import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import numpy as np
import sounddevice as sd
import threading
import queue
from collections import deque
from typing import Optional
import importlib.util
import os
import sys

# --- 定数・設定 ---
SAMPLE_RATE = 2000
PLOT_HISTORY_SIZE = 2000
DEFAULT_DATA_RATE = 2000  # 初期値として使用

# --- クラス定義と色・マーカー設定 ---
CLASS_NAMES_ALL = [
    "Healthy-Big", "Water", "Low-Small", "Healthy-Small",
    "Low-Big", "High-Small", "High-Big"
]
DISPLAY_ORDER = [
    "High-Big", "High-Small", "Low-Big", "Low-Small",
    "Healthy-Big", "Healthy-Small"
]
CLASS_COLORS = {
    "Healthy-Big": "#98FB98", "Water": "#87CEEB", "Low-Small": "#FFD700",
    "Healthy-Small": "#9ACD32", "Low-Big": "#FFA500", "High-Small": "#FF0000",
    "High-Big": "#DDA0DD"
}
CLASS_MARKERS = {
    "Healthy-Big": "o", "Water": "v", "Low-Small": "^",
    "Healthy-Small": "o", "Low-Big": "^", "High-Small": "x", "High-Big": "x"
}

# --- Audio System ---
output_stream: Optional[sd.OutputStream] = None
current_note_data = np.array([], dtype=np.float32)
playback_position = 0
data_lock = threading.Lock()

def audio_callback(outdata: np.ndarray, frames: int, time, status):
    global playback_position
    with data_lock:
        remaining_frames = frames
        buffer_offset = 0
        remaining_in_note = len(current_note_data) - playback_position
        
        if remaining_in_note > 0:
            frames_to_play = min(remaining_frames, remaining_in_note)
            chunk = current_note_data[playback_position : playback_position + frames_to_play]
            outdata[buffer_offset : buffer_offset + frames_to_play] = chunk.reshape(-1, 1)
            playback_position += frames_to_play
            remaining_frames -= frames_to_play
            buffer_offset += frames_to_play
        
        if remaining_frames > 0:
            outdata[buffer_offset:] = 0

def setup_audio_stream():
    global output_stream
    try:
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32', 
            latency='low', callback=audio_callback
        )
        output_stream.start()
        print("Audio stream initialized.")
    except Exception as e:
        print(f"Audio Error: {e}")

def play_tone(frequency: float):
    """モデルから呼ばれる音声再生関数"""
    global current_note_data, playback_position
    beep_duration = 0.1
    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), False, dtype=np.float32)
    note = 0.4 * np.sin(2 * np.pi * frequency * t)
    with data_lock:
        current_note_data = note
        playback_position = 0

# --- GUI Application ---
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Modular Peak Monitor System")
        try:
            self.master.state('zoomed')
        except:
            self.master.attributes('-zoomed', True)
        self.pack(fill="both", expand=True)

        # 共有ステート
        self.data_queue = queue.Queue()
        self.peak_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.processing_thread = None
        self.is_paused = False 

        # データバッファ
        self.plot_x_data = deque()
        self.plot_data_ch1 = deque()
        self.plot_data_ch2 = deque()
        self.peak_data_ch2 = {name: {"x": deque(), "y": deque()} for name in CLASS_NAMES_ALL}
        self.peak_data_ch2["Unknown"] = {"x": deque(), "y": deque()}
        self.class_counts = {name: 0 for name in CLASS_NAMES_ALL}
        self.class_counts["Unknown"] = 0
        self.seeker_var = tk.DoubleVar()
        self.total_lines = 0
        
        # --- パス管理用変数 ---
        self.model_display_name_var = tk.StringVar() 
        self.current_model_full_path = ""
        
        self.data_display_name_var = tk.StringVar()
        self.current_data_full_path = ""
        
        # --- レート管理用変数 ---
        self.data_rate_var = tk.StringVar(value=str(DEFAULT_DATA_RATE))
        self.current_data_rate = DEFAULT_DATA_RATE
        
        # --- Aman's GUI variables ---
        self.aman_gui_var = tk.BooleanVar(value=False)
        self.aman_gui_process = None
        
        # --- Daniel's approach variables ---
        self.daniel_approach_var = tk.BooleanVar(value=False)
        self.daniel_peak_count_ch1 = 0
        self.daniel_peak_count_ch2 = 0

        self.create_widgets()
        self.setup_plot()
        
        # アニメーションループ
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, init_func=self.init_app_state,
            interval=50, blit=False
        )

    def create_widgets(self):
        # --- Top Control Bar ---
        self.top_bar = tk.Frame(self, bg="#f0f0f0", bd=1, relief="raised")
        self.top_bar.pack(side="top", fill="x", padx=5, pady=5)
        
        # --- 1. Model Selection ---
        tk.Label(self.top_bar, text="Model:", bg="#f0f0f0", font=("Arial", 9, "bold")).pack(side="left", padx=(5, 2))
        self.model_entry = ttk.Entry(self.top_bar, textvariable=self.model_display_name_var, width=20, state="readonly")
        self.model_entry.pack(side="left", padx=2)
        ttk.Button(self.top_bar, text="...", width=3, command=self.browse_model_file).pack(side="left", padx=2)
        
        tk.Label(self.top_bar, text=" | ", bg="#f0f0f0").pack(side="left", padx=5)

        # --- 2. Data Selection ---
        tk.Label(self.top_bar, text="Data:", bg="#f0f0f0", font=("Arial", 9, "bold")).pack(side="left", padx=(5, 2))
        self.data_entry = ttk.Entry(self.top_bar, textvariable=self.data_display_name_var, width=20, state="readonly")
        self.data_entry.pack(side="left", padx=2)
        ttk.Button(self.top_bar, text="...", width=3, command=self.browse_data_file).pack(side="left", padx=2)
        
        tk.Label(self.top_bar, text=" | ", bg="#f0f0f0").pack(side="left", padx=5)

        # --- 3. Rate Input (New) ---
        tk.Label(self.top_bar, text="Rate (Hz):", bg="#f0f0f0", font=("Arial", 9, "bold")).pack(side="left", padx=(5, 2))
        self.rate_entry = ttk.Entry(self.top_bar, textvariable=self.data_rate_var, width=8)
        self.rate_entry.pack(side="left", padx=2)

        # --- 4. Load Button ---
        self.load_btn = ttk.Button(self.top_bar, text="Load", command=self.load_model)
        self.load_btn.pack(side="left", padx=20)
        
        tk.Label(self.top_bar, text=" | ", bg="#f0f0f0").pack(side="left", padx=5)
        
        # --- 5. Aman's Approach Checkbox ---
        self.aman_checkbox = tk.Checkbutton(
            self.top_bar, 
            text="Aman's approach", 
            variable=self.aman_gui_var,
            command=self.toggle_aman_gui,
            bg="#f0f0f0",
            font=("Arial", 9, "bold")
        )
        self.aman_checkbox.pack(side="left", padx=10)
        
        # --- 6. Daniel's Approach Checkbox ---
        self.daniel_checkbox = tk.Checkbutton(
            self.top_bar, 
            text="Daniel's approach", 
            variable=self.daniel_approach_var,
            bg="#f0f0f0",
            font=("Arial", 9, "bold")
        )
        self.daniel_checkbox.pack(side="left", padx=10)

        # --- Time Display ---
        self.time_label = tk.Label(self.top_bar, text="Time: 00:00", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        self.time_label.pack(side="right", padx=10)

        # --- 初期表示 ---
        self.model_display_name_var.set("Choose File...")
        self.data_display_name_var.set("Choose File...")
        self.current_model_full_path = ""
        self.current_data_full_path = ""

        # --- Bottom Counts ---
        self.bottom_area = tk.Frame(self)
        self.bottom_area.pack(side="bottom", fill="x", pady=10)
        self.counter_inner = ttk.Frame(self.bottom_area)
        self.counter_inner.pack(side="top")
        
        self.stat_labels = {}
        for name in DISPLAY_ORDER:
            bg = CLASS_COLORS.get(name, "#ddd")
            f = tk.Frame(self.counter_inner, bg=bg, bd=2, relief="solid")
            f.pack(side="left", padx=5)
            l = tk.Label(f, text=f"{name}\n0", font=("Helvetica", 9, "bold"), bg=bg, width=12)
            l.pack(padx=2, pady=2)
            self.stat_labels[name] = l
        
        # --- Daniel's Approach Counters ---
        self.daniel_counter_frame = ttk.Frame(self.bottom_area)
        self.daniel_counter_frame.pack(side="top", pady=(10, 0))
        
        # ADC1 Counter
        daniel_bg1 = "#B0E0E6"  # Light blue
        f1 = tk.Frame(self.daniel_counter_frame, bg=daniel_bg1, bd=2, relief="solid")
        f1.pack(side="left", padx=5)
        self.daniel_label_ch1 = tk.Label(f1, text="Daniel ADC1\n0", font=("Helvetica", 10, "bold"), bg=daniel_bg1, width=14)
        self.daniel_label_ch1.pack(padx=2, pady=2)
        
        # ADC2 Counter
        daniel_bg2 = "#ADD8E6"  # Lighter blue
        f2 = tk.Frame(self.daniel_counter_frame, bg=daniel_bg2, bd=2, relief="solid")
        f2.pack(side="left", padx=5)
        self.daniel_label_ch2 = tk.Label(f2, text="Daniel ADC2\n0", font=("Helvetica", 10, "bold"), bg=daniel_bg2, width=14)
        self.daniel_label_ch2.pack(padx=2, pady=2)

        # --- Plot Area ---
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 0))

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        
        self.tools_container = ttk.Frame(self.plot_frame)
        self.tools_container.pack(side="bottom", fill="x")
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.tools_container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        
        self.center_btn_frame = ttk.Frame(self.tools_container)
        self.center_btn_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Play/Pauseボタン (初期は無効)
        self.play_pause_button = ttk.Button(
            self.center_btn_frame, text="Play", command=self.toggle_play_pause, state="disabled"
        )
        self.play_pause_button.pack()

        self.status_label = ttk.Label(self.plot_frame, text="Select Model and Data files, set Rate, then click Load.", font=("Helvetica", 10))
        self.status_label.pack(side="bottom", fill="x", padx=10)

        self.seeker = ttk.Scale(self.plot_frame, orient=tk.HORIZONTAL, variable=self.seeker_var, command=self.on_seeker_drag, state="disabled")
        self.seeker.pack(side="bottom", fill="x")

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_plot(self):
        # 1. 線の初期化
        self.line_ch1, = self.ax.plot([], [], color='gray', alpha=0.5, linewidth=1, label="Ch1")
        self.line_ch2, = self.ax.plot([], [], color='black', alpha=0.8, linewidth=1, label="Ch2")
        
        # 2. ピークプロットの初期化
        self.peak_plots = {}
        for cls in CLASS_NAMES_ALL:
            c = CLASS_COLORS.get(cls, "black")
            m = CLASS_MARKERS.get(cls, "o")
            ms = 8
            mew = 3.0 if m == 'x' else 1.0
            
            self.peak_plots[cls], = self.ax.plot([], [], marker=m, color=c, 
                                                 linestyle='None', markersize=ms, 
                                                 markeredgewidth=mew, 
                                                 label=cls)
            
        self.peak_plots["Unknown"], = self.ax.plot([], [], 'ko', markersize=6, alpha=0.5, label="Unknown")
        
        # --- 凡例の順序指定 ---
        legend_order_names = [
            "High-Big", "High-Small", 
            "Low-Big", "Low-Small", 
            "Healthy-Big", "Healthy-Small", 
            "Water", "Unknown"
        ]
        
        handles = []
        labels = []
        
        for name in legend_order_names:
            if name in self.peak_plots:
                handles.append(self.peak_plots[name])
                labels.append(name)
        
        handles.append(self.line_ch1)
        labels.append("channel1")
        handles.append(self.line_ch2)
        labels.append("channel2")

        self.ax.legend(handles, labels, loc='upper right', fontsize='x-small', ncol=5, framealpha=0.9)
        
        self.ax.grid(True, linestyle=':')
        self.ax.set_ylim(450, 2000)
        self.ax.set_xlim(0, PLOT_HISTORY_SIZE)
        
        # 軸設定 (初期値ベース)
        self.ax.xaxis.set_major_locator(mticker.MultipleLocator(base=self.current_data_rate))
        self.ax.xaxis.set_major_formatter(mticker.NullFormatter())
        self.ax.xaxis.set_minor_locator(mticker.MultipleLocator(base=self.current_data_rate / 10))

    def browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
            initialdir="."
        )
        if filename:
            self.current_model_full_path = filename
            self.model_display_name_var.set(os.path.basename(filename))

    def browse_data_file(self):
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("All Files", "*.*"), ("Text Files", "*.txt"), ("CSV Files", "*.csv")],
            initialdir="."
        )
        if filename:
            self.current_data_full_path = filename
            self.data_display_name_var.set(os.path.basename(filename))

    def load_model(self):
        """モデルとデータをロードして待機状態にする"""
        model_path = self.current_model_full_path
        data_path = self.current_data_full_path
        
        if not model_path or self.model_display_name_var.get() == "Choose File...":
            self.status_label.config(text="Error: Please select a model file.")
            return
        if not data_path or self.data_display_name_var.get() == "Choose File...":
            self.status_label.config(text="Error: Please select a data file.")
            return
        
        # レートの取得と検証
        try:
            input_rate = int(self.data_rate_var.get())
            if input_rate <= 0: raise ValueError
            self.current_data_rate = input_rate
        except ValueError:
            self.status_label.config(text="Error: Invalid Rate. Please enter a positive integer.")
            return
        
        if not os.path.exists(model_path):
            self.status_label.config(text="Error: Model file not found.")
            return
        if not os.path.exists(data_path):
            self.status_label.config(text="Error: Data file not found.")
            return

        # 既存のスレッド停止
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.pause_event.set() 
            self.processing_thread.join(timeout=2.0)
        
        # --- リセット処理 (ここで軸間隔も更新) ---
        self.init_app_state()

        # --- 初期状態の設定 (StartではなくWait状態) ---
        self.stop_event.clear()
        self.pause_event.clear() 
        self.is_paused = True
        
        self.play_pause_button.config(text="Play", state="normal")
        self.seeker.config(state="normal")
        
        if hasattr(self, 'ani'):
            self.ani.pause()

        m_name = os.path.basename(model_path)
        d_name = os.path.basename(data_path)
        self.status_label.config(text=f"Loaded: {m_name} | Data: {d_name} @ {self.current_data_rate}Hz (Ready to Play)")

        # 動的インポートとスレッド起動
        try:
            module_name = os.path.splitext(m_name)[0]
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {model_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if hasattr(module, 'ProcessingModel'):
                model_instance = module.ProcessingModel(
                    self.data_queue, self.peak_queue, 
                    self.stop_event, self.pause_event, 
                    play_tone,
                    filename=data_path,
                    data_rate=self.current_data_rate # レートを渡す
                )
                
                self.processing_thread = threading.Thread(target=model_instance.run, daemon=True)
                self.processing_thread.start()
            else:
                self.status_label.config(text="Error: 'ProcessingModel' class not found in file.")
                
        except Exception as e:
            self.status_label.config(text=f"Load Error: {e}")
            print(f"Error loading model: {e}")

    def init_app_state(self):
        """アプリケーションの状態を初期化（リセット）する"""
        # 1. データのクリア
        self.plot_x_data.clear()
        self.plot_data_ch1.clear()
        self.plot_data_ch2.clear()
        for k in self.peak_data_ch2:
            self.peak_data_ch2[k]["x"].clear()
            self.peak_data_ch2[k]["y"].clear()
            self.peak_plots[k].set_data([], [])
        
        with self.data_queue.mutex: self.data_queue.queue.clear()
        with self.peak_queue.mutex: self.peak_queue.queue.clear()

        # 2. カウンタのリセット
        self.class_counts = {k: 0 for k in self.class_counts}
        for lbl in self.stat_labels.values():
            original_text = lbl.cget('text').split('\n')[0] 
            lbl.config(text=f"{original_text}\n0")
        
        # Reset Daniel's counters
        self.daniel_peak_count_ch1 = 0
        self.daniel_peak_count_ch2 = 0
        self.daniel_label_ch1.config(text="Daniel ADC1\n0")
        self.daniel_label_ch2.config(text="Daniel ADC2\n0")

        # 3. タイマーとシークバーのリセット
        self.total_lines = 0
        self.time_label.config(text="Time: 00:00")
        self.seeker_var.set(0)
        self.seeker.config(to=1) 

        # 4. グラフ線のリセット
        self.line_ch1.set_data([], [])
        self.line_ch2.set_data([], [])
        
        # グラフ範囲と軸設定を初期化（新しいレートを反映）
        self.ax.set_xlim(0, PLOT_HISTORY_SIZE)
        
        # 軸の目盛りを動的に更新
        self.ax.xaxis.set_major_locator(mticker.MultipleLocator(base=self.current_data_rate))
        self.ax.xaxis.set_minor_locator(mticker.MultipleLocator(base=self.current_data_rate / 10))
        
        self.canvas.draw()

        return [self.line_ch1, self.line_ch2] + list(self.peak_plots.values())

    def update_plot(self, frame):
        # 1. 生データ取得
        try:
            while True:
                item = self.data_queue.get_nowait()
                if item is None: # EOF
                    self.status_label.config(text="Processing Complete.")
                    self.pause_execution()
                    return
                line_num, c1, c2 = item
                self.plot_x_data.append(line_num)
                self.plot_data_ch1.append(c1)
                self.plot_data_ch2.append(c2)
                self.total_lines = line_num
        except queue.Empty: pass

        # 2. ピークデータ取得
        try:
            while True:
                item = self.peak_queue.get_nowait()
                ch_id, px, py, label = item
                
                # Check if it's a Daniel's approach peak
                if label.startswith("Daniel-"):
                    if ch_id == 1:
                        self.daniel_peak_count_ch1 += 1
                    elif ch_id == 2:
                        self.daniel_peak_count_ch2 += 1
                elif ch_id == 2:
                    key = label if label in self.peak_data_ch2 else "Unknown"
                    self.peak_data_ch2[key]["x"].append(px)
                    self.peak_data_ch2[key]["y"].append(py)
                    
                    cnt_key = label if label in self.class_counts else "Unknown"
                    self.class_counts[cnt_key] += 1
        except queue.Empty: pass

        # 3. 描画更新
        if not self.is_paused and self.plot_x_data:
            xmax = self.plot_x_data[-1]
            xmin = xmax - PLOT_HISTORY_SIZE
            self.ax.set_xlim(xmin, xmax + (PLOT_HISTORY_SIZE * 0.05))
            if xmin > 0: self.seeker_var.set(xmin)
            
            # 時間表示更新 (動的レート使用)
            sec = self.total_lines / self.current_data_rate
            self.time_label.config(text=f"Time: {int(sec//60):02d}:{int(sec%60):02d}")
            self.seeker.config(to=max(1, self.total_lines - PLOT_HISTORY_SIZE))

        # カウント更新
        for name, lbl in self.stat_labels.items():
            if name in self.class_counts:
                lbl.config(text=f"{name}\n{self.class_counts[name]}")
        
        # Update Daniel's counters
        self.daniel_label_ch1.config(text=f"Daniel ADC1\n{self.daniel_peak_count_ch1}")
        self.daniel_label_ch2.config(text=f"Daniel ADC2\n{self.daniel_peak_count_ch2}")

        # ライン描画更新
        if self.plot_data_ch1:
            self.line_ch1.set_data(self.plot_x_data, self.plot_data_ch1)
            self.line_ch2.set_data(self.plot_x_data, self.plot_data_ch2)
            
            for cls, data in self.peak_data_ch2.items():
                self.peak_plots[cls].set_data(data["x"], data["y"])

        return [self.line_ch1, self.line_ch2] + list(self.peak_plots.values())

    def toggle_play_pause(self):
        if self.is_paused: self.resume_execution()
        else: self.pause_execution()

    def pause_execution(self):
        self.is_paused = True
        self.pause_event.clear()
        if hasattr(self, 'ani'): self.ani.pause()
        self.play_pause_button.config(text="Play")
        self.seeker.config(state="normal")

    def resume_execution(self):
        self.is_paused = False
        self.pause_event.set()
        if hasattr(self, 'ani'): self.ani.resume()
        self.play_pause_button.config(text="Pause")
        self.seeker.config(state="disabled")
    
    def on_seeker_drag(self, val):
        if self.is_paused:
            xmin = float(val)
            self.ax.set_xlim(xmin, xmin + PLOT_HISTORY_SIZE)
            self.canvas.draw()
    
    def toggle_aman_gui(self):
        """Toggle Aman's GUI window on/off"""
        if self.aman_gui_var.get():
            # Open GUI.py in a separate process
            self.open_aman_gui()
        else:
            # Close GUI.py if it's running
            self.close_aman_gui()
    
    def open_aman_gui(self):
        """Open GUI.py in a separate process"""
        import subprocess
        gui_path = os.path.join(os.path.dirname(__file__), "GUI.py")
        
        if os.path.exists(gui_path):
            try:
                # Launch GUI.py as a separate process
                self.aman_gui_process = subprocess.Popen([sys.executable, gui_path])
                print(f"Opened Aman's GUI: {gui_path}")
            except Exception as e:
                print(f"Error opening Aman's GUI: {e}")
        if self.aman_gui_var.get():
            # Open GUI.py in a separate process
            self.open_aman_gui()
        else:
            # Close GUI.py if it's running
            self.close_aman_gui()
    
    def open_aman_gui(self):
        """Open GUI.py in a separate process"""
        import subprocess
        gui_path = os.path.join(os.path.dirname(__file__), "GUI.py")
        
        if os.path.exists(gui_path):
            try:
                # Launch GUI.py as a separate process
                self.aman_gui_process = subprocess.Popen([sys.executable, gui_path])
                print(f"Opened Aman's GUI: {gui_path}")
            except Exception as e:
                print(f"Error opening Aman's GUI: {e}")
                self.aman_gui_var.set(False)
        else:
            print(f"GUI.py not found at: {gui_path}")
            self.aman_gui_var.set(False)
    
    def close_aman_gui(self):
        """Close GUI.py process if it's running"""
        if self.aman_gui_process and self.aman_gui_process.poll() is None:
            try:
                self.aman_gui_process.terminate()
                print("Closed Aman's GUI")
            except Exception as e:
                print(f"Error closing Aman's GUI: {e}")
        self.aman_gui_process = None

def on_closing(root, app):
    app.stop_event.set()
    app.pause_event.set()
    
    # Close Aman's GUI if running
    if hasattr(app, 'aman_gui_process') and app.aman_gui_process and app.aman_gui_process.poll() is None:
        try:
            app.aman_gui_process.terminate()
        except:
            pass
    
    if output_stream:
        output_stream.stop()
        output_stream.close()
    root.quit()
    root.destroy()

def main():
    setup_audio_stream()
    root = tk.Tk()
    app = Application(master=root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, app))
    root.mainloop()

if __name__ == "__main__":
    main()