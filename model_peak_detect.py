import time
import numpy as np
import json
import os
from scipy.signal import find_peaks

# --- 設定 (ロジック固有) ---
THRESHOLD_FILE = 'peak_thresholds.json' 
CHUNK_SIZE = 20

class ProcessingModel:
    """
    ピーク検出とクラス分類を行うロジッククラス
    """
    def __init__(self, data_queue, peak_queue, stop_event, pause_event, play_sound_callback, filename, data_rate):
        # data_rate 引数を追加
        self.data_queue = data_queue
        self.peak_queue = peak_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.play_sound_callback = play_sound_callback
        self.filename = filename
        self.data_rate = data_rate  # 受け取ったレートを保存
        
        self.thresholds = self.load_thresholds()
        
    def load_thresholds(self):
        if not os.path.exists(THRESHOLD_FILE):
            print(f"Warning: {THRESHOLD_FILE} not found.")
            return None
        try:
            with open(THRESHOLD_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return None

    def classify_peak_by_prominence(self, prominence_val):
        if self.thresholds is None:
            return "Unknown"
        
        boundaries = self.thresholds['boundaries']
        names = self.thresholds['class_names']
        
        for i, boundary in enumerate(boundaries):
            if prominence_val < boundary:
                return names[i]
        return names[-1]

    def run(self):
        """メイン処理ループ"""
        print(f"Model Started: Reading {self.filename} at {self.data_rate} Hz")
        
        total_lines_processed = 0
        
        # 動的なレートを使用
        time_per_line = 1.0 / self.data_rate 
        target_duration_per_chunk = time_per_line * CHUNK_SIZE

        detect_buffer_ch1 = []
        detect_buffer_ch2 = []
        buffer_global_offset = 0
        
        DETECT_WINDOW = 1000
        OVERLAP = 200
        
        last_peak_idx_ch1 = -1
        last_peak_idx_ch2 = -1
        last_water_idx = -1
        last_water_suppression_window = 0
        
        REBOUND_SCALING_FACTOR = 100
        WATER_PROMINENCE = 15
        PEAK_PROMINENCE = 30
        PEAK_DISTANCE = 15

        try:
            with open(self.filename, 'r') as f:
                # Skip header row if present
                first_line = f.readline()
                if first_line:
                    # Check if it's a header (contains 'adc' or doesn't start with a number)
                    first_char = first_line.strip().lstrip(',')[0] if first_line.strip() else ''
                    if 'adc' in first_line.lower() or (first_char and not first_char.lstrip('-').replace('.','',1).isdigit()):
                        print(f"Skipping header: {first_line.strip()}")
                    else:
                        # If it's not a header, rewind to process it as data
                        f.seek(0)
                
                while not self.stop_event.is_set():
                    self.pause_event.wait()
                    
                    chunk_start_time = time.perf_counter()
                    lines_read_in_chunk = 0
                    
                    for _ in range(CHUNK_SIZE):
                        if self.stop_event.is_set(): break
                        
                        line = f.readline()
                        if not line:
                            print("Processed to the end of the file.")
                            self.stop_event.set()
                            break
                        
                        lines_read_in_chunk += 1
                        total_lines_processed += 1
                        
                        try:
                            parts = line.strip().split(',')
                            
                            # Handle both 2-column (adc1,adc2) and 3-column (index,adc1,adc2) formats
                            if len(parts) >= 2:
                                # Take the last 2 columns as adc1 and adc2
                                ch1_str = parts[-2]
                                ch2_str = parts[-1]
                                
                                # Convert to float first (handles decimals), then to int
                                ch1 = int(float(ch1_str))
                                ch2 = int(float(ch2_str))
                                
                                self.data_queue.put((total_lines_processed, ch1, ch2))

                                detect_buffer_ch1.append(ch1)
                                detect_buffer_ch2.append(ch2)

                                if len(detect_buffer_ch1) >= DETECT_WINDOW:
                                    y_ch1 = np.array(detect_buffer_ch1)
                                    y_ch2 = np.array(detect_buffer_ch2)
                                    
                                    # --- Ch1 Peak Detection ---
                                    peaks1, _ = find_peaks(y_ch1, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
                                    for p_idx in peaks1:
                                        global_idx = buffer_global_offset + p_idx
                                        if global_idx > last_peak_idx_ch1:
                                            peak_val = y_ch1[p_idx]
                                            
                                            freq = 880.0 + (peak_val - 500) * 2
                                            self.play_sound_callback(freq)
                                            
                                            self.peak_queue.put((1, global_idx, peak_val, "Ch1-Peak"))
                                            last_peak_idx_ch1 = global_idx

                                    # --- Ch2 Classification ---
                                    y_ch2_inv = -1 * y_ch2
                                    valleys, props_w = find_peaks(y_ch2_inv, prominence=WATER_PROMINENCE, distance=PEAK_DISTANCE)
                                    
                                    if len(valleys) > 0:
                                        proms_w = props_w['prominences']
                                        for i, v_idx in enumerate(valleys):
                                            global_idx = buffer_global_offset + v_idx
                                            if global_idx > last_water_idx:
                                                peak_val = y_ch2[v_idx]
                                                self.peak_queue.put((2, global_idx, peak_val, "Water"))
                                                last_water_idx = global_idx
                                                last_water_suppression_window = int(proms_w[i] * REBOUND_SCALING_FACTOR)

                                    peaks2, props2 = find_peaks(y_ch2, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
                                    if len(peaks2) > 0:
                                        proms2 = props2['prominences']
                                        for i, p_idx in enumerate(peaks2):
                                            global_idx = buffer_global_offset + p_idx
                                            if p_idx > (len(y_ch2) - 50): continue

                                            if global_idx > last_peak_idx_ch2:
                                                peak_val = y_ch2[p_idx]
                                                cls_name = self.classify_peak_by_prominence(proms2[i])

                                                is_rebound = False
                                                if last_water_idx != -1:
                                                    dist = global_idx - last_water_idx
                                                    if 0 < dist < last_water_suppression_window:
                                                        if cls_name != "High-Big":
                                                            is_rebound = True
                                                
                                                if not is_rebound:
                                                    self.peak_queue.put((2, global_idx, peak_val, cls_name))
                                                    last_peak_idx_ch2 = global_idx

                                    keep = OVERLAP
                                    detect_buffer_ch1 = detect_buffer_ch1[-keep:]
                                    detect_buffer_ch2 = detect_buffer_ch2[-keep:]
                                    buffer_global_offset += (DETECT_WINDOW - keep)

                        except (ValueError, IndexError):
                            pass
                    
                    if lines_read_in_chunk < CHUNK_SIZE:
                        break

                    elapsed = time.perf_counter() - chunk_start_time
                    sleep_time = target_duration_per_chunk - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except FileNotFoundError:
            print(f"Error: Data file '{self.filename}' not found.")
            self.stop_event.set()
        
        self.data_queue.put(None)
        print("Model thread finished.")