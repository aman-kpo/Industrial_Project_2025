import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import winsound


# ------------------------------------------
# UTILITIES (Same as your original functions)
# ------------------------------------------

def histogram_threshold(dev, percentile=99.7):
    dev = dev[np.isfinite(dev)]
    return np.nanpercentile(dev, percentile)

def is_true_peak(df_local, idx, order_cols, base_col, signal_col, min_rise):
    row = df_local.loc[idx]

    for i in range(len(order_cols) - 1):
        if not (row[order_cols[i]] > row[order_cols[i+1]]):
            return False

    base_val = row[base_col]
    if not (row[signal_col] > base_val + min_rise):
        return False

    prev_idx = idx - 1
    next_idx = idx + 1

    prev_val = df_local.at[prev_idx, signal_col] if prev_idx in df_local.index else -np.inf
    next_val = df_local.at[next_idx, signal_col] if next_idx in df_local.index else -np.inf

    return row[signal_col] > prev_val and row[signal_col] > next_val

def reduce_to_single_peak(df_local, signal_col, flag_col):
    peak_indices = df_local.index[df_local[flag_col]].tolist()
    if not peak_indices:
        return df_local

    groups, group = [], [peak_indices[0]]

    for idx in peak_indices[1:]:
        if idx == group[-1] + 1:
            group.append(idx)
        else:
            groups.append(group)
            group = [idx]
    groups.append(group)

    df_local[flag_col] = False

    for g in groups:
        max_idx = df_local.loc[g, signal_col].idxmax()
        df_local.at[max_idx, flag_col] = True

    return df_local


# ------------------------------------------
# CHUNK PROCESSING VERSION
# ------------------------------------------

def process_csv_in_chunks(path, callback_update, callback_done):
    """
    Runs in background thread.
    Calls callback_update(df_partial) after each chunk.
    Calls callback_done(df_final) when finished.
    Only processes ADC1 channel now.
    """
    chunksize = 5000
    df_full = []

    # First load in chunks
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk['adc1'] = chunk['adc1'].astype('float32')
        df_full.append(chunk)
        callback_update(chunk)   # update UI

    df = pd.concat(df_full, ignore_index=True)

    # Calculate moving averages for ADC1 only
    # ------------------------------------------------

    ma_names = ['MA001','MA002','MA003','MA005','MA008','MA013','MA021','MA034',
                'MA055','MA089','MA144','MA233','MA377','MA610','MA987']

    ms_values = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    ma_window_map = {name: int(ms / 0.5) for name, ms in zip(ma_names, ms_values)}

    base_ma_name = 'MA987'
    base_col_adc1 = f'{base_ma_name}ADC1'

    for name, win in ma_window_map.items():
        df[f'{name}ADC1'] = df['adc1'].rolling(win).mean()

    df = df.dropna().copy()

    df['dev_adc1'] = df['adc1'] - df[base_col_adc1]

    thr_adc1 = histogram_threshold(df['dev_adc1'])

    eight = [1, 2, 3, 5, 8, 13, 21, 34]
    order_cols_adc1 = [f'MA{ms:03d}ADC1' for ms in eight]

    # Store threshold and order columns for real-time calculation
    df.attrs['thr_adc1'] = thr_adc1
    df.attrs['order_cols_adc1'] = order_cols_adc1
    df.attrs['base_col_adc1'] = base_col_adc1

    callback_done(df)


# ------------------------------------------
# TKINTER GUI
# ------------------------------------------

class PeakGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Peak Detector - ADC1 Only")

        self.df = pd.DataFrame()
        self.animation = None
        self.paused = False
        self.frame = 0
        self.beeped_peaks = set()
        self.peak_count = 0
        self.beep_thread = None
        self.beep_running = False
        self.speed_multiplier = 1.0  # Default speed


        tk.Button(root, text="Upload CSV", command=self.load_file,
                  font=("Arial", 14), width=20).pack(pady=10)

        # Play/Pause Buttons
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack()

        tk.Button(ctrl_frame, text="▶ Play", command=self.play_animation,
                  font=("Arial", 12), width=10).grid(row=0, column=0, padx=5)

        tk.Button(ctrl_frame, text="⏸ Pause", command=self.pause_animation,
                  font=("Arial", 12), width=10).grid(row=0, column=1, padx=5)

        # Speed Control
        speed_frame = tk.Frame(root)
        speed_frame.pack(pady=10)

        tk.Label(speed_frame, text="Animation Speed:", font=("Arial", 11)).grid(row=0, column=0, padx=5)
        
        tk.Button(speed_frame, text="1x", command=lambda: self.set_speed(1.0),
                  font=("Arial", 10), width=5).grid(row=0, column=1, padx=2)
        tk.Button(speed_frame, text="2x", command=lambda: self.set_speed(2.0),
                  font=("Arial", 10), width=5).grid(row=0, column=2, padx=2)
        tk.Button(speed_frame, text="5x", command=lambda: self.set_speed(5.0),
                  font=("Arial", 10), width=5).grid(row=0, column=3, padx=2)
        tk.Button(speed_frame, text="10x", command=lambda: self.set_speed(10.0),
                  font=("Arial", 10), width=5).grid(row=0, column=4, padx=2)
        tk.Button(speed_frame, text="50x", command=lambda: self.set_speed(50.0),
                  font=("Arial", 10), width=5).grid(row=0, column=5, padx=2)
        tk.Button(speed_frame, text="100x", command=lambda: self.set_speed(100.0),
                  font=("Arial", 10), width=5).grid(row=0, column=6, padx=2)
        tk.Button(speed_frame, text="200x", command=lambda: self.set_speed(200.0),
                  font=("Arial", 10), width=5).grid(row=0, column=7, padx=2)
        tk.Button(speed_frame, text="500x", command=lambda: self.set_speed(500.0),
                  font=("Arial", 10), width=5).grid(row=0, column=8, padx=2)

        # Speed slider
        slider_frame = tk.Frame(root)
        slider_frame.pack()
        
        tk.Label(slider_frame, text="Fine Control:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.speed_slider = tk.Scale(slider_frame, from_=0.1, to=500.0, resolution=0.1,
                                      orient=tk.HORIZONTAL, length=400,
                                      command=self.on_slider_change)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(slider_frame, text="1.0x", font=("Arial", 10), width=8)
        self.speed_label.pack(side=tk.LEFT, padx=5)

        self.status_lbl = tk.Label(root, text="No file loaded", font=("Arial", 12))
        self.status_lbl.pack()

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    # --------------------------------------
    # FILE UPLOAD
    # --------------------------------------
    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return

        self.status_lbl.config(text="Processing... please wait")
        self.root.update_idletasks()

        # Start worker thread
        t = threading.Thread(
            target=process_csv_in_chunks,
            args=(path, self.update_partial_plot, self.finish_processing)
        )
        t.start()

    # --------------------------------------
    # CALLED DURING CHUNK PROCESSING
    # --------------------------------------
    def update_partial_plot(self, chunk):
        """Updates the plot slowly as chunks load."""
        self.df = pd.concat([self.df, chunk], ignore_index=True)

        if len(self.df) < 2000:
            return  # wait for enough data

        self.ax.clear()
        self.ax.plot(self.df["adc1"], label="ADC1", color='blue')
        self.ax.legend()
        self.ax.set_title("Loading data...")
        self.canvas.draw()

    # --------------------------------------
    # CALLED WHEN FULL PROCESSING IS DONE
    # --------------------------------------
    def finish_processing(self, df_final):
        self.df = df_final
        self.peak_count = 0

        self.status_lbl.config(text=f"Ready! ADC1 Peaks will be detected in real-time")

        self.start_beep_thread()
        self.start_animation()

    # --------------------------------------
    # BEEP THREAD FOR NON-BLOCKING SOUND
    # --------------------------------------
    def start_beep_thread(self):
        """Start a background thread for beeping"""
        self.beep_running = True
        self.beep_thread = threading.Thread(target=self.beep_worker, daemon=True)
        self.beep_thread.start()
    
    def beep_worker(self):
        """Worker thread that plays beeps"""
        import queue
        self.beep_queue = queue.Queue()
        while self.beep_running:
            try:
                # Wait for beep signal
                self.beep_queue.get(timeout=0.1)
                try:
                    winsound.Beep(1000, 120)
                except:
                    pass  # Ignore beep errors
            except queue.Empty:
                continue
    
    def trigger_beep(self):
        """Trigger a beep in the background thread"""
        if hasattr(self, 'beep_queue'):
            try:
                self.beep_queue.put(1, block=False)
            except:
                pass

    # --------------------------------------
    # ANIMATION SECTION WITH REAL-TIME PEAK DETECTION
    # --------------------------------------
    def update_frame(self, i):
        if self.paused:
            return

        # Calculate how many frames to advance based on speed
        # At high speeds, we skip frames to make it actually fast
        frames_to_advance = max(1, int(self.speed_multiplier))
        
        self.frame = min(self.frame + frames_to_advance, len(self.df) - 1)
        if self.frame >= len(self.df):
            return

        # Real-time peak detection for ALL frames we're advancing through
        # This ensures we don't miss peaks when moving fast
        thr_adc1 = self.df.attrs.get('thr_adc1', 0)
        order_cols_adc1 = self.df.attrs.get('order_cols_adc1', [])
        base_col_adc1 = self.df.attrs.get('base_col_adc1', '')
        
        # Check all frames from last position to current for peaks
        start_check = max(10, self.frame - frames_to_advance)
        for check_frame in range(start_check, self.frame + 1):
            if check_frame >= len(self.df):
                break
            idx = self.df.index[check_frame]
            
            if idx not in self.beeped_peaks:
                if is_true_peak(self.df, idx, order_cols_adc1, base_col_adc1, 'adc1', thr_adc1):
                    self.beeped_peaks.add(idx)
                    self.peak_count += 1
                    self.trigger_beep()

        self.status_lbl.config(text=f"ADC1 Peaks Detected: {self.peak_count}")

        # Plot data - Convert samples to time (sampling frequency = 5000 Hz)
        SAMPLING_FREQ = 5000.0
        x = np.arange(self.frame + 1) / SAMPLING_FREQ  # Convert to seconds
        y1 = self.df["adc1"].iloc[:self.frame + 1].values

        self.ax.clear()
        self.ax.plot(x, y1, label="ADC1", color='blue', linewidth=1)

        # Show detected peaks
        if len(self.beeped_peaks) > 0:
            peak_indices = [p for p in self.beeped_peaks if p <= self.frame]
            if peak_indices:
                peak_x = np.array([self.df.index.get_loc(p) for p in peak_indices]) / SAMPLING_FREQ
                peak_y = self.df.loc[peak_indices, "adc1"].values
                self.ax.plot(peak_x, peak_y, "ro", markersize=8, label=f"Peaks ({len(peak_indices)})")

        self.ax.legend()
        current_time = self.frame / SAMPLING_FREQ
        total_time = len(self.df) / SAMPLING_FREQ
        self.ax.set_title(f"ADC1 Signal - Speed: {self.speed_multiplier:.1f}x ({current_time:.2f}s / {total_time:.2f}s)")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("ADC1 Value")
        self.canvas.draw()

    def start_animation(self):
        self.beeped_peaks = set()
        self.peak_count = 0
        self.frame = 0  # Initialize frame counter
        if self.animation:
            self.animation.event_source.stop()

        self.paused = False
        # Use constant interval - speed is controlled by frame skipping in update_frame
        self.animation = FuncAnimation(
            self.fig, self.update_frame,
            frames=len(self.df) * 10,  # Large number to prevent early stopping
            interval=20,  # Fixed 20ms interval
            repeat=False
        )
        self.canvas.draw()

    def play_animation(self):
        self.paused = False

    def pause_animation(self):
        self.paused = True
    
    def set_speed(self, speed):
        """Set animation speed - speed is controlled by frame skipping"""
        self.speed_multiplier = speed
        self.speed_slider.set(speed)
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def on_slider_change(self, value):
        """Handle slider value change"""
        speed = float(value)
        self.set_speed(speed)


# ------------------------------------------
# RUN APP
# ------------------------------------------

root = tk.Tk()
gui = PeakGUI(root)
root.mainloop()
