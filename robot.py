import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import serial
import serial.tools.list_ports
import threading
import time

# 全域變數用於序列埠通訊
ser = None
serial_thread_running = False
serial_thread = None

# 預設參數值 (來自您的 C++ 程式碼)
DEFAULT_INITIAL_HEIGHT = 20.0
DEFAULT_LIFT_HEIGHT = 5.0 # 假設一個預設值
DEFAULT_STEP_LENGTH = 0.0
DEFAULT_SIDE_STEP_LENGTH = 0.0
DEFAULT_ROTATION_ANGLE = 0.0
DEFAULT_NUM_INTERPOLATION_POINTS = 2000
DEFAULT_MAX_ROLL_ANGLE = 10.0
DEFAULT_BALANCE_VAR = 5.0

class RobotControllerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("機器人控制器 (Robot Controller)")
        self.geometry("850x750")

        # 初始化變數
        self.initial_height_var = tk.DoubleVar(value=DEFAULT_INITIAL_HEIGHT)
        self.lift_height_var = tk.DoubleVar(value=DEFAULT_LIFT_HEIGHT)

        self.step_length_var = tk.DoubleVar(value=DEFAULT_STEP_LENGTH)
        self.side_step_length_var = tk.DoubleVar(value=DEFAULT_SIDE_STEP_LENGTH)
        self.rotation_angle_var = tk.DoubleVar(value=DEFAULT_ROTATION_ANGLE)

        self.speed_var = tk.IntVar(value=DEFAULT_NUM_INTERPOLATION_POINTS)
        self.swing_var = tk.DoubleVar(value=DEFAULT_MAX_ROLL_ANGLE)
        self.balance_var = tk.DoubleVar(value=DEFAULT_BALANCE_VAR)

        self.port_var = tk.StringVar()
        self.baud_rate_var = tk.StringVar(value="1000000")

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- 連線管理 ---
        conn_frame = ttk.LabelFrame(main_frame, text="連線管理 (Connection Management)", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="序列埠 (Port):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports: ports = ["N/A"]
        self.port_menu = ttk.Combobox(conn_frame, textvariable=self.port_var, values=ports, width=15)
        if ports[0] != "N/A": self.port_var.set(ports[0])
        self.port_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(conn_frame, text="鮑率 (Baud):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.baud_entry = ttk.Entry(conn_frame, textvariable=self.baud_rate_var, width=10)
        self.baud_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        self.connect_button = ttk.Button(conn_frame, text="連線 (Connect)", command=self._toggle_connect)
        self.connect_button.grid(row=0, column=4, padx=5, pady=5)

        self.status_label = ttk.Label(conn_frame, text="狀態: 未連線 (Status: Disconnected)")
        self.status_label.grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky=tk.W)

        # --- 核心高度設定 ---
        core_height_frame = ttk.LabelFrame(main_frame, text="核心高度設定 (Core Heights)", padding="10")
        core_height_frame.pack(fill=tk.X, pady=5)

        self._create_slider_entry(core_height_frame, "初始高度 (Initial Z - initialHeight):", self.initial_height_var, 10.0, 50.0, 0, cmd_prefix="set_initial_height")
        self._create_slider_entry(core_height_frame, "抬腳高度 (Lift Z - liftHeight):", self.lift_height_var, 0.0, 30.0, 1, cmd_prefix="set_lift_height")

        self.init_robot_button = ttk.Button(core_height_frame, text="初始化機器人 (Initialize Robot)", command=lambda: self._send_command("robot_init"))
        self.init_robot_button.grid(row=2, column=0, columnspan=4, pady=10)


        # --- 步態參數 ---
        gait_frame = ttk.LabelFrame(main_frame, text="步態參數 (Gait Parameters)", padding="10")
        gait_frame.pack(fill=tk.X, pady=5)

        self._create_slider_entry(gait_frame, "速度 (Speed - numInterpolationPoints):", self.speed_var, 500, 5000, 0, is_int=True)
        self._create_slider_entry(gait_frame, "擺幅 (Swing - MAX_ROLL_ANGLE):", self.swing_var, 0.0, 25.0, 1)
        self._create_slider_entry(gait_frame, "平衡 (Balance - balanceVar):", self.balance_var, 0.0, 15.0, 2)

        self.update_gait_button = ttk.Button(gait_frame, text="更新步態參數 (Update Gait)", command=self._send_gait_params)
        self.update_gait_button.grid(row=3, column=0, columnspan=4, pady=10)

        # --- 行走控制 ---
        walking_frame = ttk.LabelFrame(main_frame, text="行走控制 (Walking Controls)", padding="10")
        walking_frame.pack(fill=tk.X, pady=5)

        self._create_slider_entry(walking_frame, "前後步長 (X - stepLength):", self.step_length_var, -50.0, 50.0, 0)
        self._create_slider_entry(walking_frame, "左右步長 (Y - sideStepLength):", self.side_step_length_var, -30.0, 30.0, 1)
        self._create_slider_entry(walking_frame, "旋轉角度 (Theta - rotationAngle):", self.rotation_angle_var, -45.0, 45.0, 2)

        walk_button_frame = ttk.Frame(walking_frame)
        walk_button_frame.grid(row=3, column=0, columnspan=4, pady=10)
        self.start_walk_button = ttk.Button(walk_button_frame, text="開始行走 (Start Walking)", command=self._send_run_command)
        self.start_walk_button.pack(side=tk.LEFT, padx=5)
        self.stop_walk_button = ttk.Button(walk_button_frame, text="停止行走 (Stop Walking)", command=lambda: self._send_command("q"))
        self.stop_walk_button.pack(side=tk.LEFT, padx=5)


        # --- 輸出日誌 ---
        log_frame = ttk.LabelFrame(main_frame, text="輸出日誌 (Output Log)", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _create_slider_entry(self, parent, label_text, var, from_, to_, row_idx, is_int=False, cmd_prefix=None):
        ttk.Label(parent, text=label_text).grid(row=row_idx, column=0, padx=5, pady=2, sticky=tk.W)
        slider = ttk.Scale(parent, from_=from_, to=to_, variable=var, orient=tk.HORIZONTAL,
                           command=lambda val, v=var, e=None: self._update_entry_from_slider(val, v, e) if e else None) # Avoid initial call
        slider.grid(row=row_idx, column=1, padx=5, pady=2, sticky=tk.EW)

        entry_val_cmd = (self.register(lambda P, v=var, s=slider: self._validate_and_update_slider(P, v, s, from_, to_, is_int)), '%P')
        entry = ttk.Entry(parent, textvariable=var, width=8, validate='key', validatecommand=entry_val_cmd)
        entry.grid(row=row_idx, column=2, padx=5, pady=2)

        if cmd_prefix:
            update_button = ttk.Button(parent, text="更新(Update)",
                                       command=lambda p=cmd_prefix, v=var: self._send_command(f"{p},{v.get():.2f}" if not is_int else f"{p},{v.get()}"))
            update_button.grid(row=row_idx, column=3, padx=5, pady=2)

        parent.grid_columnconfigure(1, weight=1) # Make slider expand

    def _update_entry_from_slider(self, val_str, var, entry_widget_unused):
        # This is called by slider, var is already updated
        pass # Entry is directly linked to var

    def _validate_and_update_slider(self, P_value_str, var_tk, slider_widget, min_val, max_val, is_int_type):
        if P_value_str == "": # Allow empty field for typing
            return True
        try:
            if is_int_type:
                value = int(P_value_str)
            else:
                value = float(P_value_str)

            # No need to clamp here for entry, can allow temporary out-of-bounds typing
            # Value will be clamped when sending command or if slider updates it
            var_tk.set(value) # This updates the slider if it's linked
            # slider_widget.set(value) # This might cause recursion if not careful
            return True
        except ValueError:
            return False # Reject non-numeric input

    def _send_command(self, command_str):
        global ser
        if ser and ser.is_open:
            try:
                full_command = command_str + "\n"
                ser.write(full_command.encode('utf-8'))
                self._log_message(f"Sent: {full_command.strip()}")
            except Exception as e:
                self._log_message(f"Error sending command: {e}")
                messagebox.showerror("傳送錯誤 (Send Error)", f"傳送指令失敗 (Failed to send command): {e}")
        else:
            self._log_message("Error: Serial port not connected.")
            messagebox.showwarning("未連線 (Not Connected)", "序列埠未連線 (Serial port is not connected.)")

    def _send_gait_params(self):
        cmd = f"set_params,{self.speed_var.get()},{self.swing_var.get():.2f},{self.balance_var.get():.2f}"
        self._send_command(cmd)

    def _send_run_command(self):
        # Clamp values to ensure they are within reasonable limits if needed,
        # although sliders should manage this. Here, we just get current values.
        sl = self.step_length_var.get()
        rl = self.rotation_angle_var.get()
        ssl = self.side_step_length_var.get()
        cmd = f"run,{sl:.2f},{rl:.2f},{ssl:.2f}"
        self._send_command(cmd)

    def _log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _toggle_connect(self):
        global ser, serial_thread_running, serial_thread
        if ser and ser.is_open:
            # Disconnect
            serial_thread_running = False
            if serial_thread and serial_thread.is_alive():
                serial_thread.join(timeout=1)
            ser.close()
            self.status_label.config(text="狀態: 未連線 (Status: Disconnected)")
            self.connect_button.config(text="連線 (Connect)")
            self._log_message("Disconnected from serial port.")
        else:
            # Connect
            port = self.port_var.get()
            baud = self.baud_rate_var.get()
            if not port or port == "N/A":
                messagebox.showerror("連線錯誤 (Connection Error)", "請選擇一個有效的序列埠 (Please select a valid serial port).")
                return
            try:
                ser = serial.Serial(port, int(baud), timeout=1)
                self.status_label.config(text=f"狀態: 已連線至 {port} (Status: Connected to {port})")
                self.connect_button.config(text="斷線 (Disconnect)")
                self._log_message(f"Connected to {port} at {baud} baud.")

                serial_thread_running = True
                serial_thread = threading.Thread(target=self._read_from_serial, daemon=True)
                serial_thread.start()

            except Exception as e:
                ser = None
                self.status_label.config(text="狀態: 連線失敗 (Status: Connection Failed)")
                self._log_message(f"Error connecting to serial port: {e}")
                messagebox.showerror("連線錯誤 (Connection Error)", f"無法連線至序列埠 (Could not connect to serial port {port}):\n{e}")

    def _read_from_serial(self):
        global ser, serial_thread_running
        while serial_thread_running and ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    if line:
                        # Schedule the log_message call to be run in the main thread
                        self.after(0, self._log_message, f"Recv: {line}")
            except serial.SerialException: # Handle disconnects
                if serial_thread_running: # Only if we didn't intend to stop
                    self.after(0, self._handle_serial_error)
                break
            except Exception: # Catch other potential errors like decode errors
                # self.after(0, self._log_message, f"Serial read error: {e}") # Can be noisy
                pass
            time.sleep(0.05) # Reduce CPU usage

    def _handle_serial_error(self):
        # This function is called from the serial thread via `after` to safely update UI
        global ser, serial_thread_running
        if serial_thread_running : # Check if it's an unexpected disconnect
            self._log_message("Serial connection lost or error.")
            if ser: ser.close()
            ser = None
            serial_thread_running = False
            self.status_label.config(text="狀態: 連線中斷 (Status: Disconnected - Error)")
            self.connect_button.config(text="連線 (Connect)")
            messagebox.showerror("序列埠錯誤 (Serial Error)", "序列埠連線中斷或發生錯誤 (Serial port connection lost or error occurred).")


    def _on_closing(self):
        global ser, serial_thread_running
        if ser and ser.is_open:
            serial_thread_running = False
            if serial_thread and serial_thread.is_alive():
                serial_thread.join(timeout=1)
            ser.close()
        self.destroy()

if __name__ == "__main__":
    app = RobotControllerApp()
    app.mainloop()