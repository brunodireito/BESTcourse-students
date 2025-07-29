import tkinter as tk
import threading
import time
import pickle
import numpy as np
from tkinter import filedialog, messagebox
from bitalino import BITalino
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.neighbors import KNeighborsClassifier



class LieDetector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lie Detector")
        self.geometry("1920x1080")

        # Recording variables
        self.device_connected = False
        self.recording = False

        # Device and data variables
        self.recording_device = None
        self.recoding_thread = None
        self.cached_data = None

        # Lie detector variables
        self.length_of_data_for_lie_detection = 10    # Change if want
        self.model = None
        self.acquire_data_for_lie_detection = False
        self.detect_lie_data = None
        self.detect_lie_thread = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def create_widgets(self):

        # Device Info frame with border and title
        top_section = tk.Frame(self, highlightbackground="black", highlightthickness=1)
        top_section.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        title_label = tk.Label(top_section, text="Device Info", font=("Helvetica", 12, "bold"))
        title_label.pack(anchor="w", padx=5, pady=(5, 0))

        # Inner frame to hold form entries and buttons
        form_frame = tk.Frame(top_section)
        form_frame.pack(fill=tk.X, padx=10, pady=5)

        # Bitalino address
        tk.Label(form_frame, text="Bitalino address:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.address_entry = tk.Entry(form_frame, width=30)
        self.address_entry.grid(row=0, column=1, padx=5, pady=2)

        # Sampling rate
        tk.Label(form_frame, text="Sampling rate:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.sampling_entry = tk.Entry(form_frame, width=30)
        self.sampling_entry.grid(row=1, column=1, padx=5, pady=2)

        # Connect button
        self.connect_button = tk.Button(form_frame, text="Connect", command=self.connect_device)
        self.connect_button.grid(row=0, column=2, padx=5, pady=2)

        # Main frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Plot frames with borders and titles
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_canvases = []

        sensor_names = {
            0: "Respiration",
            1: "ECG",
            2: "EDA",
        }

        for i in range(3):
            # Create a frame with border and title
            section_frame = tk.Frame(plot_frame, highlightbackground="black", highlightthickness=1, bd=0)
            section_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            section_title = tk.Label(section_frame, text=f"{sensor_names[i]}", font=("Helvetica", 12, "bold"))
            section_title.pack(anchor="w", padx=5, pady=(5, 0))

            fig = Figure(figsize=(4, 2.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (mV)")
            ax.plot([], [])

            fig.tight_layout(pad=2.0)
            canvas = FigureCanvasTkAgg(fig, master=section_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.plot_canvases.append((fig, ax, canvas))

        # Right-side control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=10)

        self.load_button = tk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(pady=10)

        self.start_recording_button = tk.Button(control_frame, text="Start Recording", command=self.start_recording)
        self.start_recording_button.pack(pady=10)

        self.detect_button = tk.Button(control_frame, text="Lie Detect", command=self.lie_detect)
        self.detect_button.pack(pady=10)

        # Lights (green/red indicators)
        light_frame = tk.Frame(control_frame)
        light_frame.pack(pady=20)

        self.green_light = tk.Canvas(light_frame, width=30, height=30)
        self.green_circle = self.green_light.create_oval(5, 5, 25, 25, fill="grey")
        self.green_light.pack(pady=5)
        tk.Label(light_frame, text="Truth").pack()

        self.red_light = tk.Canvas(light_frame, width=30, height=30)
        self.red_circle = self.red_light.create_oval(5, 5, 25, 25, fill="grey")
        self.red_light.pack(pady=5)
        tk.Label(light_frame, text="Lie").pack()

    def connect_device(self):
        address = self.address_entry.get()
        sampling_rate = int(self.sampling_entry.get())

        if not address:
            messagebox.showwarning("Missing Address", "Please enter the Bitalino address.")
            return

        if not sampling_rate:
            messagebox.showwarning("Missing Sampling Rate", "Please enter the Sampling Rate.")
            return

        if sampling_rate not in {10, 100, 1000}:
            messagebox.showerror("Invalid Sampling Rate", "Please enter the one of the follwoing sampling rates: 10, 100, or 1000")
            return

        try:
            recording_device = BITalino(address)

            self.recording_device = {
                "device" : recording_device,
                "sampling_rate" : sampling_rate,
                "channels" : [0, 1, 2],
            }

            self.cached_data = {
                "Respiration": np.zeros(sampling_rate * 8),
                "ECG": np.zeros(sampling_rate * 8),
                "EDA": np.zeros(sampling_rate * 8),
            }

            success = True

        except Exception:
            messagebox.showwarning("Failed to connect", "Please retry connecting.")
            success = False

        if success:
            messagebox.showinfo("Connection", f"Connected to Bitalino at {address}")
            self.address_entry.config(state='disabled')
            self.sampling_entry.config(state='disabled')
            self.connect_button.config(state='disabled')
            self.device_connected = True
        else:
            messagebox.showerror("Connection Failed", "Failed to connect to Bitalino.")

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pkl"), ("All Files", "*.*")])

        if filepath:
            with open(filepath, 'rb') as file:
                self.model = pickle.load(file)
                messagebox.showinfo("Model", "Model Loaded!")
        
            self.load_button.config(state='disabled')


    def _detect_lie(self):

        self.acquire_data_for_lie_detection = True

        while self.acquire_data_for_lie_detection:
            time.sleep(1)

        respiration, ecg, eda = self.decompose_data(self.detect_lie_data)

        # Feature extraction
        x = self.feature_extraction(respiration, ecg, eda)

        # Detection logic
        y = self.model.predict(x)

        y = 1

        if y == 1:
            self.green_light.itemconfig(self.green_circle, fill="green")
            self.red_light.itemconfig(self.red_circle, fill="grey")
        else:
            self.green_light.itemconfig(self.green_circle, fill="grey")
            self.red_light.itemconfig(self.red_circle, fill="red")

        time.sleep(2)

        # Remove old data
        self.detect_lie_data = None
        self.green_light.itemconfig(self.green_circle, fill="grey")
        self.red_light.itemconfig(self.red_circle, fill="grey")
        self.detect_button.config(state='active')
        return


    def lie_detect(self):

        if not self.recording:
            messagebox.showerror("No data", "Please start recording.")
            return

        if not self.model:
            messagebox.showerror("No model loaded", "Please load a model first.")
            return

        self.detect_button.config(state='disabled')
        self.detect_lie_thread = threading.Thread(target=self._detect_lie)
        self.detect_lie_thread.start()


    def _cache_data(self, data, signal, sampling_rate):
        old_data = self.cached_data[signal]
        new_data = np.append(old_data, data)
        self.cached_data[signal] = new_data[-sampling_rate * 8:]

    def device_start_acquiring(self):
        device = self.recording_device["device"]
        sampling_rate = self.recording_device["sampling_rate"]
        channels = self.recording_device["channels"]

        device.start(sampling_rate, channels)
        self.recording = True
        time = 0

        while self.recording:
            raw_data = device.read(sampling_rate)

            sensor1 = raw_data[:, -3].tolist()  # Resp
            self._cache_data(sensor1, "Respiration", sampling_rate)

            sensor2 = raw_data[:, -2].tolist()  # ECG
            self._cache_data(sensor2, "ECG", sampling_rate)

            sensor3 = self.convert_units(raw_data[:, -1]).tolist()  # EDA
            self._cache_data(sensor3, "EDA", sampling_rate)

            if self.acquire_data_for_lie_detection:
                temp_data = np.vstack((sensor1, sensor2, sensor3))

                if self.detect_lie_data is None:
                    self.detect_lie_data = temp_data
                else:
                    self.detect_lie_data = np.hstack((self.detect_lie_data, temp_data))

                if self.detect_lie_data.shape[1] == sampling_rate * self.length_of_data_for_lie_detection:
                    self.acquire_data_for_lie_detection = False

            time += 1
            self.plot_update(time, sampling_rate)

        return

    def plot_update(self, time, fs):

        idx_to_signal = {
            0: "Respiration",
            1: "ECG",
            2: "EDA",
        }

        colores = {
            0: "b",
            1: "r",
            2: "y",
        }

        for i in range(3):
            y = self.cached_data[idx_to_signal[i]]
            x = np.linspace(time - 8, time, fs * 8)

            fig, ax, canvas = self.plot_canvases[i]
            ax.clear()

            ax.plot(x, y, color=colores[i])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal")

            fig.tight_layout(pad=2.0)
            canvas.draw()

    def start_recording(self):
        self.start_recording_button.config(state='disabled')
        self.recoding_thread = threading.Thread(target=self.device_start_acquiring, daemon=True)
        self.recoding_thread.start()

    def close(self):

        if self.recording:
            self.recording = False
            self.recording_device["device"].stop()
            self.recording_device["device"].close()

        self.destroy()

    @staticmethod
    def convert_units(data):
        eda = (data / 2 ** 10) * 3.3 / 0.12
        return eda

    @staticmethod
    def decompose_data(data):
        return data[0, :], data[1, :], data[2, :]

    @staticmethod
    def feature_extraction(respiration, ecg, eda):
        """
        Introduce your logic to extract features to predict a lie
        """
        X = np.zeros((1, 19)) # your output should be a 2D numpy array with all your features

        return X

if __name__ == "__main__":
    app = LieDetector()
    app.mainloop()


