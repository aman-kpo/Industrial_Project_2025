import time
import numpy as np
from collections import deque

# --- Configuration adapted from Daniel.py ---
# Moving Averages Windows in milliseconds
MA_WINDOWS_MS = [1, 3, 8, 21, 55, 144, 377, 987, 2584, 6765]
MA_KEYS = [f'MA{ms:04d}' for ms in MA_WINDOWS_MS]

# Logic Groups
CASCADE_KEYS = ['MA0001', 'MA0003', 'MA0008', 'MA0021', 'MA0055']
SLOW_KEYS = ['MA0144', 'MA0377', 'MA0987', 'MA2584', 'MA6765']
BASE_MA_KEY = 'MA0987'

# --- Channel Configurations ---
CONFIG_ADC1 = {
    'name': 'ADC1',
    'mayor_thresh': 600.0,
    'minor_offset': 60.0
}

CONFIG_ADC2 = {
    'name': 'ADC2',
    'mayor_thresh': 800.0,
    'minor_offset': 80.0
}

# --- Logic Thresholds ---
PERSISTENCE_MAYOR_MS = 7
MAX_PEAK_DURATION_MS = 70.0
WARMUP_PERIOD_MS = 8000

class GlobalStreamBuffer:
    """
    Circular buffer for maintaining recent data history with moving averages.
    """
    def __init__(self, size, ma_keys):
        self.size = size
        self.ma_keys = ma_keys
        self.data = {}
        self.data['time'] = np.zeros(size, dtype=np.float32)
        
        for ch in ['ADC1', 'ADC2']:
            self.data[f'{ch}_raw'] = np.zeros(size, dtype=np.float32)
            for k in ma_keys:
                self.data[f'{ch}_{k}'] = np.zeros(size, dtype=np.float32)
        
        self.ptr = 0
        self.total_samples = 0
        self.is_full = False
    
    def push(self, time_sec, raw1, ma1, raw2, ma2):
        """Inserts a new sample set into the circular buffer."""
        p = self.ptr
        self.data['time'][p] = time_sec
        
        self.data['ADC1_raw'][p] = raw1
        for k, v in ma1.items():
            self.data[f'ADC1_{k}'][p] = v
        
        self.data['ADC2_raw'][p] = raw2
        for k, v in ma2.items():
            self.data[f'ADC2_{k}'][p] = v
        
        self.ptr = (self.ptr + 1) % self.size
        self.total_samples += 1
        if self.ptr == 0:
            self.is_full = True
    
    def get_values_at_offset(self, channel, offset_from_now):
        """Retrieves data from 'offset_from_now' samples ago."""
        if offset_from_now > self.total_samples:
            return None
        idx = (self.ptr - offset_from_now + self.size) % self.size
        
        res = {
            'time': self.data['time'][idx],
            'raw': self.data[f'{channel}_raw'][idx]
        }
        for k in self.ma_keys:
            res[k] = self.data[f'{channel}_{k}'][idx]
        return res
    
    def check_cascade_history(self, channel, duration_samples):
        """
        Validates if the cascade condition (sorted MA values) held true
        for the specified duration.
        """
        if self.total_samples < duration_samples:
            return False
        for i in range(1, duration_samples + 1):
            idx = (self.ptr - i + self.size) % self.size
            vals = [self.data[f'{channel}_{k}'][idx] for k in CASCADE_KEYS]
            # Check if MAs are strictly ordered (descending)
            is_sorted = all(vals[j] >= vals[j+1] for j in range(len(vals)-1))
            if not is_sorted:
                return False
        return True


class ChannelProcessor:
    """
    State machine handling peak detection logic for a single channel.
    States: SEARCH -> ACTIVE -> SEARCH
    """
    def __init__(self, config, global_buffer, peak_callback):
        self.cfg = config
        self.name = config['name']
        self.buffer = global_buffer
        self.state = 'SEARCH'
        self.current_peak = None
        self.peak_callback = peak_callback  # Callback to report detected peaks
        self.peak_count = 0
    
    def process(self, time_sec, raw_val, ma_values, sample_count_global, sample_rate_ms, time_delta_sec):
        """Main step function called for every new sample."""
        warmup_samples = int(WARMUP_PERIOD_MS / sample_rate_ms)
        if sample_count_global < warmup_samples:
            return
        
        slow_vals = [ma_values[k] for k in SLOW_KEYS]
        envelope = max(slow_vals)
        base_ma = ma_values[BASE_MA_KEY]
        
        if self.state == 'SEARCH':
            self._logic_search(time_sec, ma_values, envelope, base_ma, sample_rate_ms, time_delta_sec)
        elif self.state == 'ACTIVE':
            self._logic_active(time_sec, raw_val, ma_values, envelope, time_delta_sec)
    
    def _logic_search(self, time_sec, ma_values, envelope, base_ma, sample_rate_ms, time_delta_sec):
        # Trigger Condition: MA0001 exceeds threshold
        trigger_val = ma_values['MA0001']
        is_mayor = trigger_val > self.cfg['mayor_thresh']
        is_minor = trigger_val > (base_ma + self.cfg['minor_offset'])
        
        if not (is_mayor or is_minor):
            return
        
        # Persistence Check: Cascade stability
        persistence_samples = int(PERSISTENCE_MAYOR_MS / sample_rate_ms)
        has_cascade = self.buffer.check_cascade_history(self.name, persistence_samples)
        if not has_cascade:
            return
        
        # Peak Confirmation
        peak_type = 'Mayor_Natural' if is_mayor else 'Minor'
        start_offset = persistence_samples
        start_data = self.buffer.get_values_at_offset(self.name, start_offset)
        self._start_peak(peak_type, start_data, time_delta_sec)
    
    def _start_peak(self, p_type, start_data, time_delta_sec):
        self.state = 'ACTIVE'
        self.current_peak = {
            'Channel': self.name,
            'Type': p_type,
            'Start_Time': start_data['time'],
            'Start_Values': {k: start_data[k] for k in MA_KEYS},
            'Active_Layers': {k: True for k in MA_KEYS},
            'Areas': {k: 0.0 for k in MA_KEYS},
            'Flag_Occlusion': 0,
            'Max_Amplitude': start_data['raw'],
            'End_Time': None
        }
    
    def _logic_active(self, time_sec, raw_val, ma_values, envelope, time_delta_sec):
        p = self.current_peak
        duration = time_sec - p['Start_Time']
        
        # Update Amplitude
        if raw_val > p['Max_Amplitude']:
            p['Max_Amplitude'] = raw_val
        
        # Upgrade Minor to Mayor if threshold crossed during event
        if p['Type'] == 'Minor' and ma_values['MA0001'] > self.cfg['mayor_thresh']:
            p['Type'] = 'Mayor_Natural'
        
        # 1. Occlusion Check (Max Duration Limit)
        if duration >= (MAX_PEAK_DURATION_MS / 1000.0):
            p['Flag_Occlusion'] = 1
            self._close_peak(time_sec)
            return
        
        # 2. Layer Energy Integration & Exit Logic
        for k in MA_KEYS:
            if not p['Active_Layers'][k]:
                continue
            
            val = ma_values[k]
            baseline = min(p['Start_Values'][k], envelope)
            
            if val > baseline:
                p['Areas'][k] += (val - baseline) * time_delta_sec
            
            # Exit condition for specific layer
            is_below_refs = (val < envelope) or (val < p['Start_Values'][k])
            is_in_mayor_domain = val > self.cfg['mayor_thresh']
            should_exit = is_below_refs and (not is_in_mayor_domain)
            
            if should_exit:
                p['Active_Layers'][k] = False
        
        # 3. Global Exit (All layers finished)
        if not any(p['Active_Layers'].values()):
            self._close_peak(time_sec)
    
    def _close_peak(self, current_time):
        p = self.current_peak
        p['End_Time'] = current_time
        
        # Report the peak via callback
        self.peak_count += 1
        self.peak_callback(self.name, p)
        
        self.state = 'SEARCH'
        self.current_peak = None


class ProcessingModel:
    """
    Daniel's approach for peak detection using moving average cascades.
    Compatible with the existing main.py interface.
    """
    def __init__(self, data_queue, peak_queue, stop_event, pause_event, play_sound_callback, filename, data_rate):
        self.data_queue = data_queue
        self.peak_queue = peak_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.play_sound_callback = play_sound_callback
        self.filename = filename
        self.data_rate = data_rate
        
        # Calculated constants
        self.sample_rate_ms = 1000 / self.data_rate
        self.time_delta_sec = 1.0 / self.data_rate
        
        # Calculate MA sample sizes
        self.ma_samples_map = {k: int(ms / self.sample_rate_ms) for k, ms in zip(MA_KEYS, MA_WINDOWS_MS)}
        
        # Buffer for circular storage (2 seconds)
        buffer_size_ms = 2000
        buffer_len = int(buffer_size_ms / self.sample_rate_ms)
        self.global_buffer = GlobalStreamBuffer(buffer_len, MA_KEYS)
        
        # Peak callback
        def peak_callback(channel, peak_data):
            # Send peak to queue for display
            # Format: (channel_id, x_position, y_value, label)
            ch_id = 1 if channel == 'ADC1' else 2
            label = f"Daniel-{peak_data['Type']}"
            # Use a placeholder position based on time
            x_pos = int(peak_data['Start_Time'] * self.data_rate)
            y_val = peak_data['Max_Amplitude']
            self.peak_queue.put((ch_id, x_pos, y_val, label))
        
        # Channel processors
        self.proc1 = ChannelProcessor(CONFIG_ADC1, self.global_buffer, peak_callback)
        self.proc2 = ChannelProcessor(CONFIG_ADC2, self.global_buffer, peak_callback)
        
        # Rolling buffers for MA calculation
        self.ma_buffers = {}
        for ch in ['ADC1', 'ADC2']:
            self.ma_buffers[ch] = {}
            for k in MA_KEYS:
                win_size = self.ma_samples_map[k]
                self.ma_buffers[ch][k] = deque(maxlen=win_size)
    
    def run(self):
        """Main processing loop"""
        print(f"Daniel's Model Started: Reading {self.filename} at {self.data_rate} Hz")
        
        total_lines_processed = 0
        time_per_line = 1.0 / self.data_rate
        chunk_size = 20
        target_duration_per_chunk = time_per_line * chunk_size
        
        # Column indices for adc1 and adc2 (will be determined from file)
        adc1_idx = None
        adc2_idx = None
        
        try:
            with open(self.filename, 'r') as f:
                # Read first line to detect format
                first_line = f.readline()
                if first_line:
                    parts = first_line.strip().lower().split(',')
                    
                    # Try to find adc1/adc2 in header
                    if 'adc1' in parts and 'adc2' in parts:
                        adc1_idx = parts.index('adc1')
                        adc2_idx = parts.index('adc2')
                        print(f"Detected header with adc1 at column {adc1_idx}, adc2 at column {adc2_idx}")
                    # Check if first line contains 'adc' (header) but columns aren't named exactly
                    elif any('adc' in p for p in parts):
                        # Header detected but use last 2 columns as fallback
                        print(f"Detected header but using default: last 2 columns for adc1/adc2")
                        adc1_idx = -2
                        adc2_idx = -1
                    else:
                        # No header detected - check if it's data
                        first_char = parts[0].lstrip(',').lstrip('-').replace('.','',1)
                        if first_char.isdigit():
                            # It's data, not a header - rewind and use last 2 columns
                            f.seek(0)
                            adc1_idx = -2
                            adc2_idx = -1
                            print(f"No header detected, using last 2 columns for adc1/adc2")
                        else:
                            # It's a header but no adc columns found - use last 2 columns
                            adc1_idx = -2
                            adc2_idx = -1
                            print(f"Header detected, using last 2 columns for adc1/adc2")
                
                while not self.stop_event.is_set():
                    self.pause_event.wait()
                    
                    chunk_start_time = time.perf_counter()
                    lines_read_in_chunk = 0
                    
                    for _ in range(chunk_size):
                        if self.stop_event.is_set():
                            break
                        
                        line = f.readline()
                        if not line:
                            print("Daniel's model: Processed to end of file.")
                            self.stop_event.set()
                            break
                        
                        lines_read_in_chunk += 1
                        total_lines_processed += 1
                        
                        try:
                            parts = line.strip().split(',')
                            
                            # Flexible column extraction
                            if len(parts) >= 2:
                                ch1_str = parts[adc1_idx]
                                ch2_str = parts[adc2_idx]
                                ch1 = int(float(ch1_str))
                                ch2 = int(float(ch2_str))
                                
                                # Send data to GUI
                                self.data_queue.put((total_lines_processed, ch1, ch2))
                                
                                # Calculate time
                                t_sim = total_lines_processed * self.time_delta_sec
                                
                                # Update MA buffers and calculate MAs
                                ma1 = self._calculate_mas('ADC1', ch1)
                                ma2 = self._calculate_mas('ADC2', ch2)
                                
                                # Push to global buffer
                                self.global_buffer.push(t_sim, ch1, ma1, ch2, ma2)
                                
                                # Process peak detection
                                self.proc1.process(t_sim, ch1, ma1, total_lines_processed, 
                                                 self.sample_rate_ms, self.time_delta_sec)
                                self.proc2.process(t_sim, ch2, ma2, total_lines_processed, 
                                                 self.sample_rate_ms, self.time_delta_sec)
                        
                        except (ValueError, IndexError):
                            pass
                    
                    if lines_read_in_chunk < chunk_size:
                        break
                    
                    # Timing control
                    elapsed = time.perf_counter() - chunk_start_time
                    sleep_time = target_duration_per_chunk - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except FileNotFoundError:
            print(f"Error: Data file '{self.filename}' not found.")
            self.stop_event.set()
        
        self.data_queue.put(None)
        print(f"Daniel's Model finished. ADC1 peaks: {self.proc1.peak_count}, ADC2 peaks: {self.proc2.peak_count}")
    
    def _calculate_mas(self, channel, value):
        """Calculate all moving averages for a value"""
        mas = {}
        for k in MA_KEYS:
            self.ma_buffers[channel][k].append(value)
            if len(self.ma_buffers[channel][k]) > 0:
                mas[k] = np.mean(self.ma_buffers[channel][k])
            else:
                mas[k] = 0.0
        return mas
