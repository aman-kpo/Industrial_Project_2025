import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
import time
import csv
import gc
import shutil
from collections import deque

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# --- USER CONFIGURATION (Colab Forms) ---
# Default is set to 2000 Hz as requested.
# If using the 5000 Hz file, change this value to 5000 in the side panel.
INPUT_FILENAME = "resampled_signal_2000.csv"  # @param {type:"string"}
SAMPLE_RATE_HZ = 2000  # @param {type:"integer"}

# --- SYSTEM CONSTANTS ---
# Derived calculations based on the Sample Rate
SAMPLE_RATE_MS = 1000 / SAMPLE_RATE_HZ
TIME_DELTA_SEC = 1.0 / SAMPLE_RATE_HZ

print(f"--- System Configuration ---")
print(f"Target Input File: {INPUT_FILENAME}")
print(f"Sample Rate: {SAMPLE_RATE_HZ} Hz (Delta: {TIME_DELTA_SEC:.6f} s)")

# --- DIRECTORY SETUP ---
# Creates a local output directory to avoid dependency on specific Drive folders
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- OUTPUT FILE PATHS ---
TIMESTAMP_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
FILE_PATH = os.path.join(BASE_DIR, INPUT_FILENAME)
OUT_LOG_FILE = os.path.join(OUTPUT_DIR, f'detection_log_{TIMESTAMP_STR}.csv')
OUT_DUMP_FILE = os.path.join(OUTPUT_DIR, f'segments_dump_{TIMESTAMP_STR}.csv')
OUT_STATS_FILE = os.path.join(OUTPUT_DIR, f'processing_stats_{TIMESTAMP_STR}.txt')

# --- MOVING AVERAGES (MA) ---
# Windows defined in milliseconds
MA_WINDOWS_MS = [1, 3, 8, 21, 55, 144, 377, 987, 2584, 6765]
MA_KEYS = [f'MA{ms:04d}' for ms in MA_WINDOWS_MS]
# Calculate sample size for each MA based on the current Hz
MA_SAMPLES_MAP = {k: int(ms / SAMPLE_RATE_MS) for k, ms in zip(MA_KEYS, MA_WINDOWS_MS)}

# Logic Groups
CASCADE_KEYS = ['MA0001', 'MA0003', 'MA0008', 'MA0021', 'MA0055']
SLOW_KEYS = ['MA0144', 'MA0377', 'MA0987', 'MA2584', 'MA6765']
BASE_MA_KEY = 'MA0987'

# --- LOGIC THRESHOLDS ---
BUFFER_SIZE_MS = 2000  # 2 seconds circular buffer
BUFFER_LEN = int(BUFFER_SIZE_MS / SAMPLE_RATE_MS)

PERSISTENCE_MAYOR_MS = 7
PERSISTENCE_MAYOR_SAMPLES = int(PERSISTENCE_MAYOR_MS / SAMPLE_RATE_MS)

WARMUP_PERIOD_MS = 8000
WARMUP_SAMPLES = int(WARMUP_PERIOD_MS / SAMPLE_RATE_MS)

# Recording Constraints
MAX_PEAK_DURATION_MS = 70.0  # Force exit (occlusion) logic
RECORDING_WINDOW_MS = 200.0  # Total duration to save
SNIPPET_PRE_MS = 15.0        # Context before event

# Plotting Constants
PLOT_PRE_START_MS = 15.0
PLOT_POST_START_MS = 85.0
PLOT_DPI = 200
PLOT_FIGSIZE = (20, 12)

# --- CHANNEL SPECIFICS ---
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

# ==========================================
# 2. DATA STRUCTURES
# ==========================================

class GlobalStreamBuffer:
    """
    Unified circular buffer holding raw and processed data for both stereo channels.
    Allows for historical lookups and snapshot extraction.
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
        if self.ptr == 0: self.is_full = True

    def get_values_at_offset(self, channel, offset_from_now):
        """Retrieves data from 'offset_from_now' samples ago."""
        if offset_from_now > self.total_samples: return None
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
        if self.total_samples < duration_samples: return False
        for i in range(1, duration_samples + 1):
            idx = (self.ptr - i + self.size) % self.size
            vals = [self.data[f'{channel}_{k}'][idx] for k in CASCADE_KEYS]
            # Check if MAs are strictly ordered (ascending/descending logic)
            is_sorted = all(vals[j] >= vals[j+1] for j in range(len(vals)-1))
            if not is_sorted: return False
        return True

    def extract_stereo_snapshot(self, start_time, end_time):
        """Extracts a slice of buffer data for saving to disk."""
        mask = (self.data['time'] >= start_time) & (self.data['time'] <= end_time)
        indices = np.where(mask)[0]
        snippet = []
        for i in indices:
            row = {
                'Abs_Time_Sec': self.data['time'][i],
                'ADC1_Raw': self.data['ADC1_raw'][i],
                'ADC2_Raw': self.data['ADC2_raw'][i]
            }
            for k in self.ma_keys:
                row[f'ADC1_{k}'] = self.data[f'ADC1_{k}'][i]
                row[f'ADC2_{k}'] = self.data[f'ADC2_{k}'][i]
            snippet.append(row)
        snippet.sort(key=lambda x: x['Abs_Time_Sec'])
        return snippet


class ChannelProcessor:
    """
    State machine handling peak detection logic for a single channel.
    States: SEARCH -> ACTIVE -> (Record & Save) -> SEARCH
    """
    def __init__(self, config, id_gen, global_buffer):
        self.cfg = config
        self.name = config['name']
        self.id_gen = id_gen
        self.buffer = global_buffer
        self.state = 'SEARCH'
        self.current_peak = None
        self.pending_peaks = deque()

    def process(self, time_sec, raw_val, ma_values, sample_count_global):
        """Main step function called for every new sample."""
        self._check_pending_queue(time_sec)
        if sample_count_global < WARMUP_SAMPLES: return

        slow_vals = [ma_values[k] for k in SLOW_KEYS]
        envelope = max(slow_vals)
        base_ma = ma_values[BASE_MA_KEY]

        if self.state == 'SEARCH':
            self._logic_search(time_sec, ma_values, envelope, base_ma)
        elif self.state == 'ACTIVE':
            self._logic_active(time_sec, raw_val, ma_values, envelope)

    def _check_pending_queue(self, current_time):
        """Checks if any finished peaks are ready to be saved to disk."""
        while self.pending_peaks:
            p = self.pending_peaks[0]
            recording_end = p['Start_Time'] + (RECORDING_WINDOW_MS / 1000.0)
            if current_time >= recording_end:
                self._save_pending_peak(p, recording_end)
                self.pending_peaks.popleft()
            else:
                break

    def _save_pending_peak(self, p, recording_end):
        snip_start = p['Start_Time'] - (SNIPPET_PRE_MS / 1000.0)
        snippet = self.buffer.extract_stereo_snapshot(snip_start, recording_end)
        save_peak_data(p, snippet)

    def _logic_search(self, time_sec, ma_values, envelope, base_ma):
        # Trigger Condition: MA0001 exceeds threshold
        trigger_val = ma_values['MA0001']
        is_mayor = trigger_val > self.cfg['mayor_thresh']
        is_minor = trigger_val > (base_ma + self.cfg['minor_offset'])

        if not (is_mayor or is_minor): return

        # Persistence Check: Cascade stability
        has_cascade = self.buffer.check_cascade_history(self.name, PERSISTENCE_MAYOR_SAMPLES)
        if not has_cascade: return

        # Peak Confirmation
        peak_type = 'Mayor_Natural' if is_mayor else 'Minor'
        start_offset = PERSISTENCE_MAYOR_SAMPLES # Backtrack slightly to capture onset
        start_data = self.buffer.get_values_at_offset(self.name, start_offset)
        self._start_peak(peak_type, start_data)

    def _start_peak(self, p_type, start_data):
        self.state = 'ACTIVE'
        self.current_peak = {
            'Peak_Event_ID': next(self.id_gen),
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

    def _logic_active(self, time_sec, raw_val, ma_values, envelope):
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
            if not p['Active_Layers'][k]: continue

            val = ma_values[k]
            baseline = min(p['Start_Values'][k], envelope)

            if val > baseline:
                p['Areas'][k] += (val - baseline) * TIME_DELTA_SEC

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
        self.pending_peaks.append(p)
        self.state = 'SEARCH'
        self.current_peak = None

# ==========================================
# 3. IO & UTILITY FUNCTIONS
# ==========================================

def id_gen_func():
    """Generator for unique peak IDs."""
    i = 0
    while True:
        i += 1
        yield i
GLOBAL_ID_GEN = id_gen_func()

def init_csvs():
    """Initializes output CSV files with headers."""
    with open(OUT_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Peak_Event_ID', 'Channel', 'Type', 'Start_Time_Abs', 'End_Time_Abs',
            'Duration_ms', 'Max_Amplitude', 'Flag_Occlusion',
            'Area_MA0001', 'Area_MA0055'
        ]
        writer.writerow(header)

    with open(OUT_DUMP_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Peak_Event_ID', 'Abs_Time_Sec', 'Rel_Time_ms', 'ADC1_Raw', 'ADC2_Raw']
        for k in MA_KEYS: header.append(f'ADC1_{k}')
        for k in MA_KEYS: header.append(f'ADC2_{k}')
        writer.writerow(header)

    with open(OUT_STATS_FILE, 'w') as f:
        f.write("Batch_ID,System_Time,Processing_Latency_microsecond\n")

def save_peak_data(p, snippet):
    """Writes peak metadata and raw snippet data to CSVs."""
    duration_ms = (p['End_Time'] - p['Start_Time']) * 1000.0

    row = [
        p['Peak_Event_ID'],
        p['Channel'],
        p['Type'],
        f"{p['Start_Time']:.4f}",
        f"{p['End_Time']:.4f}",
        f"{duration_ms:.1f}",
        f"{p['Max_Amplitude']:.1f}",
        p['Flag_Occlusion'],
        f"{p['Areas']['MA0001']:.2f}",
        f"{p['Areas']['MA0055']:.2f}"
    ]

    with open(OUT_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    rows_dump = []
    for s in snippet:
        rel = (s['Abs_Time_Sec'] - p['Start_Time']) * 1000.0
        r = [
            p['Peak_Event_ID'],
            f"{s['Abs_Time_Sec']:.4f}",
            f"{rel:.1f}",
            f"{s['ADC1_Raw']:.1f}",
            f"{s['ADC2_Raw']:.1f}"
        ]
        for k in MA_KEYS: r.append(f"{s[f'ADC1_{k}']:.1f}")
        for k in MA_KEYS: r.append(f"{s[f'ADC2_{k}']:.1f}")
        rows_dump.append(r)

    with open(OUT_DUMP_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows_dump)

# ==========================================
# 4. MAIN EXECUTION ENGINE
# ==========================================

def run_stream_simulation():
    print("\n--- Starting Stream Engine ---")
    init_csvs()

    # --- INPUT FILE VALIDATION ---
    if not os.path.exists(FILE_PATH):
        print(f"\n[ERROR] Input file '{INPUT_FILENAME}' not found in the current directory.")
        print("Please upload the .csv file to the Colab files area (left sidebar).")
        print("If using a different file, update the 'INPUT_FILENAME' variable in the config.")
        return False

    print("Loading CSV feed (Float32)...")
    try:
        df_feed = pd.read_csv(FILE_PATH, dtype=np.float32)
        # Normalize column names if necessary
        df_feed = df_feed.rename(columns={'adc2': 'adc2', 'adc1': 'adc1'})
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    # --- PRE-CALCULATE MOVING AVERAGES ---
    print("Pre-calculating MAs for simulation feed...")
    feed_dict = {}
    for ch in ['ADC1', 'ADC2']:
        raw_col = 'adc1' if ch == 'ADC1' else 'adc2'
        feed_dict[f'{ch}_RAW'] = df_feed[raw_col].values
        for k, win in zip(MA_KEYS, MA_WINDOWS_MS):
            w_size = int(win / SAMPLE_RATE_MS)
            # Rolling mean
            feed_dict[f'{ch}_{k}'] = df_feed[raw_col].rolling(w_size).mean().fillna(0).astype(np.float32).values

    total_samples = len(df_feed)
    print(f"Feed ready: {total_samples} samples")

    # --- INITIALIZE PROCESSORS ---
    global_buffer = GlobalStreamBuffer(BUFFER_LEN, MA_KEYS)
    proc1 = ChannelProcessor(CONFIG_ADC1, GLOBAL_ID_GEN, global_buffer)
    proc2 = ChannelProcessor(CONFIG_ADC2, GLOBAL_ID_GEN, global_buffer)

    print("Streaming started...")
    t0 = time.time()

    adc1_raw_arr = feed_dict['ADC1_RAW']
    adc2_raw_arr = feed_dict['ADC2_RAW']

    # --- MAIN LOOP ---
    for i in range(total_samples):
        loop_start = time.time()
        t_sim = i * TIME_DELTA_SEC

        # Construct current data packet
        row1 = {k: feed_dict[f'ADC1_{k}'][i] for k in MA_KEYS}
        row2 = {k: feed_dict[f'ADC2_{k}'][i] for k in MA_KEYS}

        # Push to buffer
        global_buffer.push(t_sim, adc1_raw_arr[i], row1, adc2_raw_arr[i], row2)

        # Process logic
        proc1.process(t_sim, adc1_raw_arr[i], row1, i)
        proc2.process(t_sim, adc2_raw_arr[i], row2, i)

        # Stats logging (Latency check)
        if i % 5000 == 0 and i > 0:
            loop_end = time.time()
            lat_us = (loop_end - loop_start) * 1_000_000
            with open(OUT_STATS_FILE, 'a') as f:
                f.write(f"{i},{time.time()},{lat_us:.1f}\n")

    print(f"Simulation Done. Real-world execution time: {time.time()-t0:.2f}s")
    return True

# ==========================================
# 5. POST-PROCESSING & PLOTTING
# ==========================================

def analyze_and_plot_top_peaks_final():
    print("\n--- Starting Post-Processing Analysis ---")

    if not os.path.exists(OUT_LOG_FILE) or not os.path.exists(OUT_DUMP_FILE):
        print("Output files not found. Simulation might have failed.")
        return

    df_log = pd.read_csv(OUT_LOG_FILE)
    if df_log.empty:
        print("Log is empty. No peaks detected.")
        return

    df_dump = pd.read_csv(OUT_DUMP_FILE)

    # --- STATISTICS SUMMARY ---
    print("\n[STATISTICS SUMMARY]")
    total_mayor = len(df_log[df_log['Type'].str.contains('Mayor')])
    total_minor = len(df_log[df_log['Type'] == 'Minor'])
    count_occlusions = df_log['Flag_Occlusion'].sum()

    stats_text = (
        f"Total Peaks: {len(df_log)}\n"
        f"  - Mayor: {total_mayor}\n"
        f"  - Minor: {total_minor}\n"
        f"Peaks Occluded (Cut at {MAX_PEAK_DURATION_MS}ms): {count_occlusions}\n"
    )
    print(stats_text)

    with open(os.path.join(OUTPUT_DIR, f'final_stats_{TIMESTAMP_STR}.txt'), 'w') as f:
        f.write(stats_text)

    # --- PLOTTING LOGIC (Clustered) ---
    # Limit to first 50 peaks for safety
    candidates_df = df_log.sort_values('Start_Time_Abs').head(50)
    print(f"Generating plots for {len(candidates_df)} peaks...")

    # Cluster peaks that are close in time to reduce figure count
    clusters = []
    current_cluster = []

    for _, peak in candidates_df.iterrows():
        if not current_cluster:
            current_cluster.append(peak)
        else:
            prev_peak = current_cluster[-1]
            # Group if within 100ms
            if peak['Start_Time_Abs'] < prev_peak['Start_Time_Abs'] + 0.100:
                current_cluster.append(peak)
            else:
                clusters.append(current_cluster)
                current_cluster = [peak]
    if current_cluster: clusters.append(current_cluster)

    for cluster in clusters:
        anchor_peak = cluster[0]
        t_zero = anchor_peak['Start_Time_Abs']

        # Determine View Window
        view_start = t_zero - (PLOT_PRE_START_MS / 1000.0)
        view_end = t_zero + (PLOT_POST_START_MS / 1000.0)

        scene_data = df_dump[
            (df_dump['Abs_Time_Sec'] >= view_start) &
            (df_dump['Abs_Time_Sec'] <= view_end)
        ]
        if scene_data.empty: continue

        fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE, sharex=True, dpi=PLOT_DPI)

        for i, ch in enumerate(['ADC1', 'ADC2']):
            ax = axes[i]
            # Plot Raw & MA lines
            ax.plot(scene_data['Abs_Time_Sec'], scene_data[f'{ch}_Raw'], color='black', lw=1.5, alpha=0.8, label='Raw')
            ax.plot(scene_data['Abs_Time_Sec'], scene_data[f'{ch}_MA0001'], color='green', lw=1, label='MA0001')

            # Plot Thresholds
            cfg = CONFIG_ADC1 if ch == 'ADC1' else CONFIG_ADC2
            ax.axhline(cfg['mayor_thresh'], color='purple', ls=':', alpha=0.5, label='Mayor Thr')

            # Highlight detected regions
            peaks_in_cluster = [p for p in cluster if p['Channel'] == ch]
            for p in peaks_in_cluster:
                ax.axvline(x=p['Start_Time_Abs'], color='green', alpha=0.4, lw=2)
                ax.axvline(x=p['End_Time_Abs'], color='red', alpha=0.4, lw=2)

            ax.set_title(f"{ch} Activity", fontsize=14)
            if i == 0: ax.legend(loc='upper right', fontsize=8)

        axes[1].set_xlabel("Time (s)")
        is_occ = anchor_peak['Flag_Occlusion'] == 1
        title_color = 'red' if is_occ else 'black'
        plt.suptitle(f"Peak Event #{anchor_peak['Peak_Event_ID']} (Starts: {t_zero:.4f}s)", color=title_color, fontsize=16)

        fname = f"plot_event_{anchor_peak['Peak_Event_ID']}_{TIMESTAMP_STR}.jpg"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches='tight')
        plt.close(fig)

    print("Plotting finished.")

# ==========================================
# 6. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    success = run_stream_simulation()

    if success:
        analyze_and_plot_top_peaks_final()

        # --- AUTO-ZIP OUTPUTS ---
        print("\n--- Zipping Results ---")
        shutil.make_archive('processing_results', 'zip', OUTPUT_DIR)
        print(f"Success! Download 'processing_results.zip' from the files tab.")