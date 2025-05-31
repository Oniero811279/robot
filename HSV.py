from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
import sys
import cv2
import numpy as np
import json
import os
import serial
import time
import platform
import serial.tools.list_ports
from ultralytics import YOLO # <--- NEW: Import YOLO

# 全域變數 (原程式保留，但部分已移至類別內部管理)
# cap = cv2.VideoCapture(0) # 改由 MyWidget 初始化
# mouse_x = [0, 0] # 由 OpenCVThread 管理
# mouse_y = [0, 0] # 由 OpenCVThread 管理
# press_count = 0    # 由 OpenCVThread 管理
inputData = 0      # 此變數在原程式中未被使用


class SerialCommunication:
    # Baudrate 預設值與原程式相同
    def __init__(self, port=None, baudrate=1000000): # <-- 修改：允許 port 初始化為 None
        self.ser = None
        self.port = port # <-- 新增：儲存當前連接的埠號
        self.baudrate = baudrate # <-- 新增：儲存鮑率
        self.response_buffer = "" # 確保 response_buffer 被初始化

        if port: # 如果提供了 port，則嘗試連接
            self.connect(port, baudrate)

    def connect(self, port, baudrate):
        """嘗試連接到指定的序列埠"""
        if self.ser and self.ser.is_open:
            self.close() # 如果已有連線，先關閉

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            self.port = port # 更新當前埠號
            self.baudrate = baudrate # 更新鮑率
            print(f"Serial connection established on {port} at {baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error establishing serial connection on {port}: {e}")
            self.ser = None
            return False
        except Exception as e:
            print(f"An unexpected error occurred with serial connection on {port}: {e}")
            self.ser = None
            return False

    def send_command(self, command_code,params=None):
        if self.ser and self.ser.is_open:
            param_str = ""
            if params:
                param_str = "," + ",".join(map(str, params)) # 將參數列表轉換為逗號分隔的字串

            full_command = f"<{command_code}{param_str}>\n"
            try:
                self.ser.write(full_command.encode('ascii')) # 使用 ascii 編碼
                print(f"Sent protocol command: {full_command.strip()}") # 去掉換行符再印出
            except Exception as e:
                print(f"Error sending protocol command: {e}")
        else:
            print("Serial connection not established for protocol command.")


    def read_response(self, timeout_s=0.5):
        if not (self.ser and self.ser.is_open):
            return None
        
        start_time = time.time()
        line = ""
        try:
            while (time.time() - start_time) < timeout_s:
                if self.ser.in_waiting > 0:
                    byte = self.ser.read(1)
                    try:
                        char = byte.decode('ascii')
                        if char == '\n':
                            full_line = self.response_buffer + line
                            self.response_buffer = "" 
                            return full_line.strip()
                        else:
                            line += char
                    except UnicodeDecodeError:
                        pass
                else:
                    time.sleep(0.01)
            self.response_buffer += line 
            if self.response_buffer:
                pass
            return None

        except Exception as e:
            print(f"Error reading serial response: {e}")
            self.response_buffer = "" 
            return None

    def close(self):
        if self.ser and self.ser.is_open:
            current_port = self.port 
            self.ser.close()
            print(f"Serial connection closed on {current_port if current_port else 'N/A'}.")
        self.ser = None 

    @staticmethod
    def get_available_ports():
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        print(f"Available serial ports: {available_ports}")
        return available_ports

    @staticmethod
    def get_os_specific_default_port():
        os_type = platform.system().lower()
        if "windows" in os_type:
            return "COM8" 
        elif "linux" in os_type:
            return "/dev/ttyUSB0"
        return None


class ColorProfile:
    def __init__(self, name, c1_min=0, c1_max=179, c2_min=0, c2_max=255, c3_min=0, c3_max=255, color_hex="#FFFFFF"):
        self.name = name
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2_min = c2_min
        self.c2_max = c2_max
        self.c3_min = c3_min
        self.c3_max = c3_max
        self.color_hex = color_hex
        self.active = True

    def to_dict(self):
        return {
            "name": self.name,
            "c1_min": self.c1_min, "c1_max": self.c1_max,
            "c2_min": self.c2_min, "c2_max": self.c2_max,
            "c3_min": self.c3_min, "c3_max": self.c3_max,
            "color_hex": self.color_hex,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data):
        profile = cls(
            name=data.get("name", "Unnamed"),
            c1_min=data.get("c1_min", 0),
            c1_max=data.get("c1_max", 179),
            c2_min=data.get("c2_min", 0),
            c2_max=data.get("c2_max", 255),
            c3_min=data.get("c3_min", 0),
            c3_max=data.get("c3_max", 255),
            color_hex=data.get("color_hex", "#FFFFFF")
        )
        profile.active = data.get("active", True)
        return profile


class OpenCVThread(QThread):
    update_frame = pyqtSignal(QImage) 

    def __init__(self, widget_ref, cap_ref): 
        super().__init__()
        self.widget = widget_ref
        self.cap = cap_ref 
        self.running = True
        self.mouse_press_count = 0 
        self.mouse_x_coords = [0, 0] 
        self.mouse_y_coords = [0, 0] 
        self.min_tracking_area = 100 
        self.perform_single_detection = False
        self.perform_yolo_detection = False # <--- NEW: Flag for YOLO detection
        self.yolo_model = None # <--- NEW: Placeholder for YOLO model

        try: # <--- NEW: Load YOLO model
            model_path = 'best.pt' # Assumes best.pt is in the same directory
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print(f"YOLOv8 model '{model_path}' loaded successfully.")
            else:
                print(f"Error: YOLOv8 model file '{model_path}' not found.")
                self.yolo_model = None
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.yolo_model = None


    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取影像幀 (OpenCVThread)")
                self.msleep(30) 
                continue
            
            brightness = self.widget.brightness_val
            contrast_alpha = self.widget.contrast_val / 100.0
            
            try:
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_alpha, beta=brightness)
            except cv2.error as e:
                print(f"OpenCV 亮度/對比度調整錯誤: {e}")
                adjusted_frame = frame.copy() 


            if self.mouse_press_count == 1 and self.widget.current_profile_index >= 0:
                self.process_color_selection(frame) 
                self.mouse_press_count = 0 

            processed_display_frame = self.process_image_for_display(adjusted_frame)

            qimg = self.convert_cv_to_qimage(processed_display_frame)
            if qimg: 
                 self.update_frame.emit(qimg)
            self.msleep(10) 

    def process_color_selection(self, original_frame): 
        x1_widget, x2_widget = sorted([self.mouse_x_coords[0], self.mouse_x_coords[1]])
        y1_widget, y2_widget = sorted([self.mouse_y_coords[0], self.mouse_y_coords[1]])

        label_geom = self.widget.label.geometry() 
        label_pos_in_widget = self.widget.label.mapTo(self.widget, QPoint(0,0)) 

        x1_label = max(0, x1_widget - label_pos_in_widget.x())
        y1_label = max(0, y1_widget - label_pos_in_widget.y())
        x2_label = min(label_geom.width(), x2_widget - label_pos_in_widget.x())
        y2_label = min(label_geom.height(), y2_widget - label_pos_in_widget.y())
        
        if x1_label < x2_label and y1_label < y2_label:
            frame_h, frame_w = original_frame.shape[:2]
            label_w, label_h = label_geom.width(), label_geom.height()

            if label_w == 0 or label_h == 0: return 

            scale_x = frame_w / label_w
            scale_y = frame_h / label_h

            x1_frame = int(x1_label * scale_x)
            y1_frame = int(y1_label * scale_y)
            x2_frame = int(x2_label * scale_x)
            y2_frame = int(y2_label * scale_y)

            if x1_frame < x2_frame and y1_frame < y2_frame:
                crop_img = original_frame[y1_frame:y2_frame, x1_frame:x2_frame]
                if crop_img.size > 0:
                    (c1m, c2m, c3m), (c1x, c2x, c3x) = self.get_color_range_from_selection(crop_img, self.widget.current_color_space)
                    if c1m is not None: 
                        self.widget.update_profile_channels(c1m, c1x, c2m, c2x, c3m, c3x)


    def get_color_range_from_selection(self, crop_img, color_space):
        color_range_sensitivity = 15 
        
        target_crop = None
        ch_configs = self.widget.get_channel_configs(color_space) 

        if color_space == 'HSV': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
        elif color_space == 'YCrCb': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'HLS': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HLS)
        else: target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV) 

        if target_crop.size == 0: return (None,None,None), (None,None,None)

        avg_per_channel = np.average(target_crop, axis=(0, 1))
        
        mins = np.zeros(3, dtype=int)
        maxs = np.zeros(3, dtype=int)

        for i in range(3):
            min_val, max_val = ch_configs['ranges'][i]
            mins[i] = np.clip(avg_per_channel[i] - color_range_sensitivity, min_val, max_val).astype(int)
            maxs[i] = np.clip(avg_per_channel[i] + color_range_sensitivity, min_val, max_val).astype(int)
            if mins[i] >= maxs[i]: 
                if maxs[i] < max_val: maxs[i] = mins[i] + 1
                elif mins[i] > min_val: mins[i] = maxs[i] -1
                else: 
                    pass 

        return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])

    def perform_detection_action(self, detected_objects, frame_shape):
        if not detected_objects:
            return

        frame_height, frame_width = frame_shape[:2]
        primary_target = max(detected_objects, key=lambda obj: obj['area'])
        cx = primary_target['cx']
        frame_center_x = frame_width // 2
        dead_zone = 20  

        action_description = ""
        command_to_send = None

        if cx < frame_center_x - dead_zone:
            command_to_send = "run,-15,15,0"
            action_description = f"Detected '{primary_target['profile_name']}' on the LEFT. Sending command: {command_to_send}"
        elif cx > frame_center_x + dead_zone:
            command_to_send = "run,15,-15,0"
            action_description = f"Detected '{primary_target['profile_name']}' on the RIGHT. Sending command: {command_to_send}"
        else:
            command_to_send = "run,30,0,0" 
            action_description = f"Detected '{primary_target['profile_name']}' in the CENTER. Sending command: {command_to_send}"

        if command_to_send:
            print(action_description)
            self.widget.serial_comm.send_command(command_to_send)

    def process_image_for_display(self, adjusted_frame): #取代原 process_frame
        # <--- NEW: YOLO Detection Logic --->
        if self.perform_yolo_detection and self.yolo_model:
            yolo_results = self.yolo_model(adjusted_frame, verbose=False) 
            annotated_frame = yolo_results[0].plot() 

            # <--- MODIFIED: Save frame if objects are detected --->
            if len(yolo_results[0].boxes) > 0: # 檢查是否有偵測到任何邊界框 (bounding boxes)
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # 確保檔名不包含可能導致問題的特殊字元，雖然時間戳記通常是安全的
                    filename = f"yolo_detection_{timestamp}.png"
                    
                    # 決定儲存路徑 (目前為執行腳本的目錄)
                    # 您可以更改 'save_dir' 來指定儲存到特定子資料夾
                    save_dir = "." # 代表目前工作目錄
                    if not os.path.exists(save_dir) and save_dir != ".":
                        os.makedirs(save_dir) # 如果指定了子資料夾且不存在，則建立它
                    
                    save_path = os.path.join(save_dir, filename)
                    
                    cv2.imwrite(save_path, annotated_frame) # annotated_frame 是 BGR 格式
                    print(f"YOLO 辨識結果已儲存至: {os.path.abspath(save_path)}")
                    # 您可以在這裡發送一個訊號到主視窗，顯示儲存成功的訊息
                    # 例如: self.widget.show_status_message(f"影像已儲存: {filename}")
                except Exception as e:
                    print(f"儲存 YOLO 辨識結果時發生錯誤: {e}")
            else:
                # 即使觸發了YOLO偵測，但如果沒有偵測到物體，可以選擇不儲存或提示
                print("YOLO 偵測已觸發，但未發現可儲存的物體。")
            # <--- END MODIFIED Save frame logic --->

            self.perform_yolo_detection = False # 重置旗標
            return annotated_frame # 返回帶有 YOLO 標註的畫面
        # <--- END NEW YOLO Detection Logic --->

        # Original processing logic if YOLO is not triggered
        current_cs = self.widget.current_color_space
        if current_cs == 'HSV': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        elif current_cs == 'LAB': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2LAB)
        elif current_cs == 'YCrCb': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2YCrCb)
        elif current_cs == 'HLS': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HLS)
        else: converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV) # Default

        frame_height, frame_width = adjusted_frame.shape[:2]
        output_display_frame = adjusted_frame.copy() 
        all_masks_combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
        detected_objects_for_tracking = []

        for profile in self.widget.color_profiles:
            if not profile.active:
                continue
                
            lower_bound = np.array([profile.c1_min, profile.c2_min, profile.c3_min], dtype=np.uint8)
            upper_bound = np.array([profile.c1_max, profile.c2_max, profile.c3_max], dtype=np.uint8)
            
            individual_mask = cv2.inRange(converted_frame, lower_bound, upper_bound)
            
            kernel = np.ones((5,5), np.uint8)
            individual_mask = cv2.morphologyEx(individual_mask, cv2.MORPH_OPEN, kernel)
            individual_mask = cv2.morphologyEx(individual_mask, cv2.MORPH_CLOSE, kernel)
            
            all_masks_combined = cv2.bitwise_or(all_masks_combined, individual_mask)
            
            contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_contour_for_profile = None
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_tracking_area and area > max_area:
                    max_area = area
                    largest_contour_for_profile = contour
            
            if largest_contour_for_profile is not None:
                M = cv2.moments(largest_contour_for_profile)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(largest_contour_for_profile)
                    
                    detected_objects_for_tracking.append({
                        "cx": cx, "cy": cy, "w": w, "h": h, "area": max_area, 
                        "profile_name": profile.name, "profile_color_hex": profile.color_hex
                    })
                    
                    if self.widget.display_mode == 'original_with_highlights':
                        q_color = QColor(profile.color_hex)
                        bgr_color = (q_color.blue(), q_color.green(), q_color.red())
                        cv2.rectangle(output_display_frame, (x, y), (x + w, y + h), bgr_color, 2)
                        cv2.circle(output_display_frame, (cx, cy), 7, bgr_color, -1)
                        cv2.putText(output_display_frame, profile.name, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        if detected_objects_for_tracking:
            if self.widget.tracking_center:
                primary_target = max(detected_objects_for_tracking, key=lambda obj: obj['area'])
                cx, w_obj, h_obj = primary_target['cx'], primary_target['w'], primary_target['h']
                frame_center_x = adjusted_frame.shape[1] // 2
                dead_zone = 20
                command_to_send = None

                if cx < frame_center_x - dead_zone:
                    command_to_send = "run,15,15,0" 
                elif cx > frame_center_x + dead_zone:
                    command_to_send = "run,15,-15,0"
                else: 
                    if w_obj > 150 or h_obj > 150:
                        command_to_send = "run,-15,0,0" 
                    elif w_obj > 100 or h_obj > 100:
                        command_to_send = "run,-10,0,0" 
                    else:
                        command_to_send = "run,30,0,0"  

                if command_to_send:
                    self.widget.serial_comm.send_command(command_to_send)

                cv2.line(output_display_frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 0, 0), 2)
                q_color_target = QColor(primary_target['profile_color_hex'])
                bgr_color_target = (q_color_target.blue(), q_color_target.green(), q_color_target.red())
                cv2.putText(output_display_frame, f"Tracking: {primary_target['profile_name']}", (cx - w_obj//2, primary_target['cy'] - h_obj//2 -10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color_target, 2)

            elif self.perform_single_detection:
                self.perform_detection_action(detected_objects_for_tracking, adjusted_frame.shape)

        if self.perform_single_detection:
            self.perform_single_detection = False

        if self.widget.display_mode == 'masked':
            final_result_frame = cv2.bitwise_and(adjusted_frame, adjusted_frame, mask=all_masks_combined)
            if self.widget.tracking_center and detected_objects_for_tracking:
                primary_target = max(detected_objects_for_tracking, key=lambda obj: obj['area'])
                cx_trk = primary_target['cx']
                frame_center_x_trk = frame_width // 2
                cv2.line(final_result_frame, (frame_center_x_trk, 0), (frame_center_x_trk, frame_height), (255, 0, 0), 2)
                q_color_target_trk = QColor(primary_target['profile_color_hex'])
                bgr_color_target_trk = (q_color_target_trk.blue(), q_color_target_trk.green(), q_color_target_trk.red())
                cv2.putText(final_result_frame, f"Tracking: {primary_target['profile_name']}", (cx_trk - primary_target['w']//2, primary_target['cy'] - primary_target['h']//2 -10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color_target_trk, 2)
        else: 
            final_result_frame = output_display_frame
            
        return final_result_frame


    def convert_cv_to_qimage(self, cv_img): 
        if cv_img is None: return None
        try:
            target_w = self.widget.label.width()
            target_h = self.widget.label.height()

            if target_w <= 0 or target_h <=0 : 
                return None

            if cv_img.size == 0:
                return None

            resized_img = cv2.resize(cv_img, (target_w, target_h))

            if len(resized_img.shape) == 3: 
                # YOLO's plot() function returns BGR, so convert to RGB for QImage
                if self.perform_yolo_detection: # Check if this was a yolo frame (already reset, so this condition might not be ideal)
                                               # Actually, YOLO results.plot() returns a BGR numpy array
                    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                else: # Original logic for non-YOLO frames
                    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif len(resized_img.shape) == 2: 
                h, w = resized_img.shape
                bytes_per_line = w
                return QImage(resized_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            return None
        except Exception as e:
            print(f"Error in convert_cv_to_qimage: {e}")
            return None


    def stop(self): 
        self.running = False
        self.wait()

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.color_profiles = []
        self.current_profile_index = -1
        
        self.current_color_space = 'HSV'
        self.supported_color_spaces = ['HSV', 'LAB', 'YCrCb', 'HLS'] 
        
        self.brightness_val = 0
        self.contrast_val = 100
        
        self.display_mode = 'masked'
        self.tracking_center = False
        
        self.serial_comm = SerialCommunication(baudrate=1000000)
        self.available_serial_ports = SerialCommunication.get_available_ports()
        self.selected_serial_port = None

        if self.available_serial_ports:
            os_default_port = SerialCommunication.get_os_specific_default_port()
            port_to_try = None

            if os_default_port and os_default_port in self.available_serial_ports:
                port_to_try = os_default_port
            else:
                port_to_try = self.available_serial_ports[0]
            
            if self.serial_comm.connect(port_to_try, self.serial_comm.baudrate):
                self.selected_serial_port = port_to_try
                print(f"Auto-connected to serial port: {self.selected_serial_port}")
            else:
                QtWidgets.QMessageBox.warning(self, "序列埠錯誤", f"無法自動連接至 {port_to_try}。請手動選擇。")
        else:
            QtWidgets.QMessageBox.warning(self, "序列埠錯誤", "未偵測到任何序列埠！")
            print("No serial ports detected.")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "攝影機錯誤", "無法開啟攝影機！")
            self.cap = None # Ensure cap is None if not opened
        
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        
        if self.cap: # Only create thread if cap is valid
            self.opencv_thread = OpenCVThread(self, self.cap)
        else:
            self.opencv_thread = None # Set thread to None if cap failed

        self.slider_labels = {}
        self.slider_value_labels = {}

        self.initUI()

        self.add_new_profile("預設")
        if self.color_profiles:
            self.select_profile_by_index(0)
        
        if self.opencv_thread: # Only connect and start if thread exists
            self.opencv_thread.update_frame.connect(self.update_label)
            self.opencv_thread.start()
        else:
            self.label.setText("攝影機未連接或初始化失敗")
        self.center_window()

    def center_window(self):
        """將視窗置於螢幕中央"""
        try:
            # 獲取視窗的框架幾何資訊 (包含標題列)
            frame_gm = self.frameGeometry()
            # 獲取螢幕可用區域的中心點 (不包含任務列等)
            screen_center_pt = QtWidgets.QDesktopWidget().availableGeometry().center()
            # 將視窗框架的中心點移動到螢幕的中心點
            frame_gm.moveCenter(screen_center_pt)
            # 移動視窗的左上角到計算後的位置
            self.move(frame_gm.topLeft())
        except Exception as e:
            print(f"無法將視窗置中: {e}")

    def get_channel_configs(self, color_space_name=None):
        if color_space_name is None:
            color_space_name = self.current_color_space
        
        if color_space_name == 'HSV':
            return {'names': ['H', 'S', 'V'], 'ranges': [(0, 179), (0, 255), (0, 255)]}
        elif color_space_name == 'LAB':
            return {'names': ['L', 'A', 'B'], 'ranges': [(0, 255), (0, 255), (0, 255)]}
        elif color_space_name == 'YCrCb':
            return {'names': ['Y', 'Cr', 'Cb'], 'ranges': [(0, 255), (0, 255), (0, 255)]}
        elif color_space_name == 'HLS':
            return {'names': ['H', 'L', 'S'], 'ranges': [(0, 179), (0, 255), (0, 255)]}
        else: 
            return {'names': ['H', 'S', 'V'], 'ranges': [(0, 179), (0, 255), (0, 255)]}


    def initUI(self):
        self.setWindowTitle('WeAreMinions - YOLO Capable') # <--- MODIFIED Window Title
        self.resize(1200, 750)

        main_layout = QtWidgets.QHBoxLayout(self)
        
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(10)

        profile_group = QtWidgets.QGroupBox("Color Profiles")
        profile_layout = QtWidgets.QVBoxLayout()
        self.profile_list = QtWidgets.QListWidget()
        self.profile_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.profile_list.itemClicked.connect(self.select_profile_from_list_item)
        profile_layout.addWidget(self.profile_list)
        
        profile_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self.add_profile_dialog)
        profile_btn_layout.addWidget(self.btn_add)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_current_profile)
        profile_btn_layout.addWidget(self.btn_remove)
        self.btn_toggle_active = QtWidgets.QPushButton("Toggle Active")
        self.btn_toggle_active.clicked.connect(self.toggle_current_profile_active)
        profile_btn_layout.addWidget(self.btn_toggle_active)
        profile_layout.addLayout(profile_btn_layout)
        profile_group.setLayout(profile_layout)
        left_panel.addWidget(profile_group)


        serial_port_group = QtWidgets.QGroupBox("Serial Port Selection")
        serial_port_layout = QtWidgets.QVBoxLayout()
        self.serial_port_radio_group = QtWidgets.QButtonGroup(self)

        if not self.available_serial_ports:
            no_ports_label = QtWidgets.QLabel("未找到序列埠 (No serial ports found)")
            serial_port_layout.addWidget(no_ports_label)
        else:
            for i, port_name in enumerate(self.available_serial_ports):
                rb = QtWidgets.QRadioButton(port_name)
                rb.setProperty("port_name", port_name)
                if port_name == self.selected_serial_port:
                    rb.setChecked(True)
                rb.toggled.connect(self.on_serial_port_changed)
                serial_port_layout.addWidget(rb)
                self.serial_port_radio_group.addButton(rb, i)
        
        serial_port_group.setLayout(serial_port_layout)
        left_panel.addWidget(serial_port_group)

        cs_group = QtWidgets.QGroupBox("Color Space")
        cs_layout = QtWidgets.QVBoxLayout()
        self.current_cs_label = QtWidgets.QLabel(f"Current: {self.current_color_space}")
        cs_layout.addWidget(self.current_cs_label)
        btn_change_cs = QtWidgets.QPushButton("Change Color Space...")
        btn_change_cs.clicked.connect(self.show_color_space_dialog)
        cs_layout.addWidget(btn_change_cs)
        cs_group.setLayout(cs_layout)
        left_panel.addWidget(cs_group)

        slider_group = QtWidgets.QGroupBox("Channel Controls")
        slider_main_layout = QtWidgets.QVBoxLayout()

        self.slider_c1_min, self.slider_labels['c1_min'], self.slider_value_labels['c1_min'] = self.create_slider_pair("C1", 0, 179, slider_main_layout)
        self.slider_c1_max = self.slider_c1_min.property("companion_slider")

        self.slider_c2_min, self.slider_labels['c2_min'], self.slider_value_labels['c2_min'] = self.create_slider_pair("C2", 0, 255, slider_main_layout)
        self.slider_c2_max = self.slider_c2_min.property("companion_slider")
        
        self.slider_c3_min, self.slider_labels['c3_min'], self.slider_value_labels['c3_min'] = self.create_slider_pair("C3", 0, 255, slider_main_layout)
        self.slider_c3_max = self.slider_c3_min.property("companion_slider")

        self.slider_c1_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c2_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c3_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c1_max.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c2_max.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c3_max.valueChanged.connect(self.update_current_profile_from_sliders)

        self.update_sliders_for_color_space()

        slider_group.setLayout(slider_main_layout)
        left_panel.addWidget(slider_group)

        img_adj_group = QtWidgets.QGroupBox("Image Adjustments")
        img_adj_layout = QtWidgets.QVBoxLayout()
        
        brightness_label_widget = QtWidgets.QLabel("Brightness: 0")
        img_adj_layout.addWidget(brightness_label_widget)
        self.slider_brightness = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(self.brightness_val)
        self.slider_brightness.valueChanged.connect(lambda v, lbl=brightness_label_widget: (
            self.brightness_changed(v), lbl.setText(f"Brightness: {v}")
        ))
        img_adj_layout.addWidget(self.slider_brightness)

        contrast_label_widget = QtWidgets.QLabel(f"Contrast: {self.contrast_val/100.0:.2f}x")
        img_adj_layout.addWidget(contrast_label_widget)
        self.slider_contrast = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 200)
        self.slider_contrast.setValue(self.contrast_val)
        self.slider_contrast.valueChanged.connect(lambda v, lbl=contrast_label_widget: (
            self.contrast_changed(v), lbl.setText(f"Contrast: {v/100.0:.2f}x")
        ))
        img_adj_layout.addWidget(self.slider_contrast)
        img_adj_group.setLayout(img_adj_layout)
        left_panel.addWidget(img_adj_group)


        instructions = QtWidgets.QLabel("Click to select area")
        instructions.setWordWrap(True)
        left_panel.addWidget(instructions)
        
        file_group = QtWidgets.QGroupBox("File Operations")
        file_layout = QtWidgets.QVBoxLayout()
        self.input_field = QtWidgets.QLineEdit(self)
        self.input_field.setPlaceholderText("Enter filename")
        file_layout.addWidget(self.input_field)
        io_layout = QtWidgets.QHBoxLayout()
        self.button_save = QtWidgets.QPushButton("Save All Profiles")
        self.button_save.clicked.connect(self.saveData)
        io_layout.addWidget(self.button_save)
        self.button_load = QtWidgets.QPushButton("Load Profiles")
        self.button_load.clicked.connect(self.loadData)
        io_layout.addWidget(self.button_load)
        file_layout.addLayout(io_layout)
        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)
        
        left_panel.addStretch(1)

        main_layout.addLayout(left_panel, 1)
        
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.label, stretch=1)
        
        options_group = QtWidgets.QGroupBox("Display & Tracking")
        options_layout = QtWidgets.QHBoxLayout()
        
        self.radio_masked = QtWidgets.QRadioButton("Masked")
        self.radio_original = QtWidgets.QRadioButton("Original+Highlights")
        
        if self.display_mode == 'masked':
            self.radio_masked.setChecked(True)
        else:
            self.radio_original.setChecked(True)
            
        self.radio_masked.clicked.connect(self.change_display_mode)
        self.radio_original.clicked.connect(self.change_display_mode)
        options_layout.addWidget(self.radio_masked)
        options_layout.addWidget(self.radio_original)
        
        self.btn_track_center = QtWidgets.QPushButton("Center Tracking")
        self.btn_track_center.setCheckable(True)
        self.btn_track_center.setChecked(self.tracking_center)
        self.btn_track_center.clicked.connect(self.toggle_center_tracking)
        options_layout.addWidget(self.btn_track_center)

        self.btn_detect_once = QtWidgets.QPushButton("Detect and Act")
        self.btn_detect_once.clicked.connect(self.trigger_single_detection)
        options_layout.addWidget(self.btn_detect_once)

        # <--- NEW YOLO BUTTON --->
        self.btn_yolo_detect = QtWidgets.QPushButton("YOLO Detect Once")
        self.btn_yolo_detect.clicked.connect(self.trigger_yolo_detection_once)
        options_layout.addWidget(self.btn_yolo_detect)
        # <--- END NEW YOLO BUTTON --->

        options_group.setLayout(options_layout)
        right_panel.addWidget(options_group)
        
        main_layout.addLayout(right_panel, 2)

    # <--- NEW: Method to trigger YOLO detection in the thread --->
    def trigger_yolo_detection_once(self):
        if self.opencv_thread and self.opencv_thread.yolo_model:
            print("Triggering single YOLO detection.")
            self.opencv_thread.perform_yolo_detection = True
        elif self.opencv_thread and not self.opencv_thread.yolo_model:
            QtWidgets.QMessageBox.warning(self, "YOLO Model Error", "YOLOv8 model (best.pt) is not loaded. Cannot perform detection.")
            print("YOLOv8 model not loaded. Cannot trigger detection.")
        elif not self.opencv_thread:
            QtWidgets.QMessageBox.warning(self, "Camera Error", "Camera/OpenCV thread not running. Cannot perform detection.")
            print("OpenCV thread not available for YOLO detection.")

    def trigger_single_detection(self):
        if self.opencv_thread:
            if not any(p.active for p in self.color_profiles):
                QtWidgets.QMessageBox.warning(self, "Warning", "No active color profiles to detect.")
                return
            print("Triggering single color detection.")
            self.opencv_thread.perform_single_detection = True

    def create_slider_pair(self, base_name, min_val_default, max_val_default, parent_layout):
        min_label_text = f"{base_name} Min:"
        min_slider_container = QtWidgets.QHBoxLayout()
        
        actual_min_label = QtWidgets.QLabel(min_label_text)
        self.slider_labels[f'{base_name.lower()}_min_prefix'] = actual_min_label
        min_slider_container.addWidget(actual_min_label)

        min_slider = QtWidgets.QSlider(Qt.Horizontal)
        min_slider.setRange(min_val_default, max_val_default)
        min_slider_container.addWidget(min_slider)
        
        min_value_label = QtWidgets.QLabel(str(min_slider.value()))
        min_slider.valueChanged.connect(lambda v, lbl=min_value_label, s=min_slider: (
            lbl.setText(str(v)),
            self.update_current_profile_from_sliders()
        ))
        min_slider_container.addWidget(min_value_label)
        self.slider_value_labels[f'{base_name.lower()}_min'] = min_value_label

        parent_layout.addLayout(min_slider_container)

        max_label_text = f"{base_name} Max:"
        max_slider_container = QtWidgets.QHBoxLayout()

        actual_max_label = QtWidgets.QLabel(max_label_text)
        self.slider_labels[f'{base_name.lower()}_max_prefix'] = actual_max_label
        max_slider_container.addWidget(actual_max_label)

        max_slider = QtWidgets.QSlider(Qt.Horizontal)
        max_slider.setRange(min_val_default, max_val_default)
        max_slider.setValue(max_val_default)
        max_slider_container.addWidget(max_slider)

        max_value_label = QtWidgets.QLabel(str(max_slider.value()))
        max_slider.valueChanged.connect(lambda v, lbl=max_value_label, s=max_slider: (
            lbl.setText(str(v)),
            self.update_current_profile_from_sliders()
        ))
        max_slider_container.addWidget(max_value_label)
        self.slider_value_labels[f'{base_name.lower()}_max'] = max_value_label

        parent_layout.addLayout(max_slider_container)

        min_slider.setProperty("companion_slider", max_slider)
        max_slider.setProperty("companion_slider", min_slider)
        
        return min_slider, actual_min_label, min_value_label


    def brightness_changed(self, value):
        self.brightness_val = value

    def contrast_changed(self, value):
        self.contrast_val = value


    def update_sliders_for_color_space(self):
        configs = self.get_channel_configs()
        ch_names = configs['names']
        ch_ranges = configs['ranges']

        sliders = {
            'c1': (self.slider_c1_min, self.slider_c1_max),
            'c2': (self.slider_c2_min, self.slider_c2_max),
            'c3': (self.slider_c3_min, self.slider_c3_max)
        }

        for i, key_base in enumerate(['c1', 'c2', 'c3']):
            min_slider, max_slider = sliders[key_base]
            
            self.slider_labels[f'{key_base}_min_prefix'].setText(f"{ch_names[i]} Min:")
            self.slider_labels[f'{key_base}_max_prefix'].setText(f"{ch_names[i]} Max:")
            
            min_slider.setRange(ch_ranges[i][0], ch_ranges[i][1])
            max_slider.setRange(ch_ranges[i][0], ch_ranges[i][1])

        if self.current_profile_index != -1 and self.color_profiles:
            self.load_profile_values_to_sliders(self.color_profiles[self.current_profile_index])


    def show_color_space_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("選擇色彩空間 (Select Color Space)")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        self.cs_radio_button_group = QtWidgets.QButtonGroup(dialog)
        
        for i, space_name in enumerate(self.supported_color_spaces):
            rb = QtWidgets.QRadioButton(space_name)
            if space_name == self.current_color_space:
                rb.setChecked(True)
            layout.addWidget(rb)
            self.cs_radio_button_group.addButton(rb, i)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        
        def accept_selection():
            selected_id = self.cs_radio_button_group.checkedId()
            if selected_id != -1:
                new_space = self.supported_color_spaces[selected_id]
                if new_space != self.current_color_space:
                    self.current_color_space = new_space
                    self.current_cs_label.setText(f"當前: {self.current_color_space}")
                    self.update_sliders_for_color_space()
            dialog.accept()

        button_box.accepted.connect(accept_selection)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.exec_()


    def add_new_profile(self, name):
        q_color = QColor(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color_hex = q_color.name()
        
        configs = self.get_channel_configs()
        
        new_profile = ColorProfile(
            name=name,
            c1_min=configs['ranges'][0][0], c1_max=configs['ranges'][0][1],
            c2_min=configs['ranges'][1][0], c2_max=configs['ranges'][1][1],
            c3_min=configs['ranges'][2][0], c3_max=configs['ranges'][2][1],
            color_hex=color_hex
        )
        self.color_profiles.append(new_profile)
        self.update_profile_list_display()
        self.select_profile_by_index(len(self.color_profiles) - 1)
        return len(self.color_profiles) - 1


    def add_profile_dialog(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "新增顏色設定檔", "輸入設定檔名稱:")
        if ok and name:
            self.add_new_profile(name)


    def remove_current_profile(self):
        if self.current_profile_index >= 0:
            if len(self.color_profiles) <= 1 and self.current_profile_index == 0 :
                 QtWidgets.QMessageBox.warning(self, "警告", "至少需要保留一個顏色設定檔。")
                 return

            del self.color_profiles[self.current_profile_index]
            self.update_profile_list_display()
            if self.color_profiles:
                new_index = max(0, min(self.current_profile_index, len(self.color_profiles) - 1))
                self.select_profile_by_index(new_index)
            else:
                self.current_profile_index = -1
                self.clear_sliders()
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "沒有選定的設定檔可以移除。")


    def toggle_current_profile_active(self):
        if self.current_profile_index >= 0 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            profile.active = not profile.active
            self.update_profile_list_display()


    def update_profile_list_display(self):
        self.profile_list.clear()
        for i, profile in enumerate(self.color_profiles):
            status = "✓" if profile.active else "✗"
            item = QtWidgets.QListWidgetItem(f"{status} {profile.name}")
            item.setData(Qt.UserRole, i)
            item.setForeground(QColor(profile.color_hex))
            if i == self.current_profile_index:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.profile_list.addItem(item)

    def select_profile_from_list_item(self, item):
        index = item.data(Qt.UserRole)
        self.select_profile_by_index(index)

    def select_profile_by_index(self, index):
        if 0 <= index < len(self.color_profiles):
            self.current_profile_index = index
            profile = self.color_profiles[index]
            self.load_profile_values_to_sliders(profile)
            self.update_profile_list_display()
        elif not self.color_profiles:
            self.current_profile_index = -1
            self.clear_sliders()
            self.update_profile_list_display()


    def load_profile_values_to_sliders(self, profile):
        sliders_map = {
            'c1': (self.slider_c1_min, self.slider_c1_max, profile.c1_min, profile.c1_max),
            'c2': (self.slider_c2_min, self.slider_c2_max, profile.c2_min, profile.c2_max),
            'c3': (self.slider_c3_min, self.slider_c3_max, profile.c3_min, profile.c3_max)
        }
        configs = self.get_channel_configs()

        for i, key in enumerate(['c1', 'c2', 'c3']):
            slider_min, slider_max, val_min, val_max = sliders_map[key]
            ch_range_min, ch_range_max = configs['ranges'][i]

            slider_min.blockSignals(True)
            slider_max.blockSignals(True)

            slider_min.setValue(np.clip(val_min, ch_range_min, ch_range_max))
            slider_max.setValue(np.clip(val_max, ch_range_min, ch_range_max))
            
            self.slider_value_labels[f'{key}_min'].setText(str(slider_min.value()))
            self.slider_value_labels[f'{key}_max'].setText(str(slider_max.value()))

            slider_min.blockSignals(False)
            slider_max.blockSignals(False)
            
    def clear_sliders(self):
        configs = self.get_channel_configs()

        for i in range(3):
            key_base = f'c{i+1}'
            min_slider = getattr(self, f"slider_{key_base}_min")
            max_slider = getattr(self, f"slider_{key_base}_max")
            ch_range_min, ch_range_max = configs['ranges'][i]

            min_slider.blockSignals(True)
            max_slider.blockSignals(True)

            min_slider.setValue(ch_range_min)
            max_slider.setValue(ch_range_max)
            self.slider_value_labels[f'{key_base}_min'].setText(str(ch_range_min))
            self.slider_value_labels[f'{key_base}_max'].setText(str(ch_range_max))

            min_slider.blockSignals(False)
            max_slider.blockSignals(False)


    def update_current_profile_from_sliders(self):
        if self.current_profile_index != -1 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            
            profile.c1_min = self.slider_c1_min.value()
            profile.c1_max = self.slider_c1_max.value()
            profile.c2_min = self.slider_c2_min.value()
            profile.c2_max = self.slider_c2_max.value()
            profile.c3_min = self.slider_c3_min.value()
            profile.c3_max = self.slider_c3_max.value()

            if profile.c1_min > profile.c1_max: profile.c1_min = profile.c1_max
            if profile.c2_min > profile.c2_max: profile.c2_min = profile.c2_max
            if profile.c3_min > profile.c3_max: profile.c3_min = profile.c3_max


    def update_profile_channels(self, c1m, c1x, c2m, c2x, c3m, c3x):
        if self.current_profile_index != -1 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            profile.c1_min, profile.c1_max = c1m, c1x
            profile.c2_min, profile.c2_max = c2m, c2x
            profile.c3_min, profile.c3_max = c3m, c3x
            self.load_profile_values_to_sliders(profile)
            self.update_profile_list_display()


    def mousePressEvent(self, event):
        if self.opencv_thread and event.buttons() == Qt.LeftButton and self.label.geometry().contains(event.pos()):
            self.opencv_thread.mouse_x_coords[0] = event.pos().x()
            self.opencv_thread.mouse_y_coords[0] = event.pos().y()


    def mouseReleaseEvent(self, event):
        if self.opencv_thread and event.button() == Qt.LeftButton and self.label.geometry().contains(event.pos()):
            self.opencv_thread.mouse_x_coords[1] = event.pos().x()
            self.opencv_thread.mouse_y_coords[1] = event.pos().y()
            if self.current_profile_index != -1 :
                 self.opencv_thread.mouse_press_count = 1
            else:
                 QtWidgets.QMessageBox.information(self, "提示", "請先新增或選取一個顏色設定檔。")


    def saveData(self):
        filename = self.input_field.text()
        if not filename:
            QtWidgets.QMessageBox.warning(self, "警告", "請輸入檔案名稱")
            return
            
        profiles_data = [profile.to_dict() for profile in self.color_profiles]
        save_data = {
            "profiles": profiles_data,
            "current_color_space": self.current_color_space,
            "brightness": self.brightness_val,
            "contrast": self.contrast_val
            }
        
        try:
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4)
            QtWidgets.QMessageBox.information(self, "成功", f"所有設定已儲存至 {filename}.json")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "錯誤", f"儲存失敗: {str(e)}")
    
    def loadData(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "載入設定檔", "",
                                                            "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                backup_profiles = self.color_profiles.copy()
                
                loaded_profiles_data = data.get("profiles", [])
                self.color_profiles = [ColorProfile.from_dict(pd) for pd in loaded_profiles_data]

                loaded_cs = data.get("current_color_space", "HSV")
                if loaded_cs in self.supported_color_spaces:
                    self.current_color_space = loaded_cs
                else:
                    self.current_color_space = "HSV"
                self.current_cs_label.setText(f"當前: {self.current_color_space}")
                self.update_sliders_for_color_space()

                self.brightness_val = data.get("brightness", 0)
                self.contrast_val = data.get("contrast", 100)
                self.slider_brightness.setValue(self.brightness_val)
                self.slider_contrast.setValue(self.contrast_val)

                self.update_profile_list_display()
                if self.color_profiles:
                    self.select_profile_by_index(0)
                else:
                    self.color_profiles = backup_profiles
                    QtWidgets.QMessageBox.information(self, "通知", "檔案中未找到有效的設定檔，或設定檔為空。")
                
                self.input_field.setText(os.path.splitext(os.path.basename(fileName))[0])
                QtWidgets.QMessageBox.information(self, "成功", f"已從 {fileName} 載入設定")

            except json.JSONDecodeError:
                QtWidgets.QMessageBox.critical(self, "錯誤", "檔案格式錯誤，無法解析 JSON。")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "錯誤", f"載入失敗: {str(e)}")


    def toggle_center_tracking(self):
        self.tracking_center = not self.tracking_center
        self.btn_track_center.setChecked(self.tracking_center)
        self.btn_track_center.setText("停止追蹤 (Stop)" if self.tracking_center else "中心追蹤 (Center Tracking)")


    def change_display_mode(self):
        self.display_mode = 'masked' if self.radio_masked.isChecked() else 'original_with_highlights'


    def update_label(self, qimg):
        if qimg and not qimg.isNull():
            self.label.setPixmap(QPixmap.fromImage(qimg))


    def on_serial_port_changed(self):
        radio_button = self.sender()
        if radio_button and radio_button.isChecked():
            new_port_name = radio_button.property("port_name")
            if new_port_name != self.selected_serial_port or not self.serial_comm.ser or not self.serial_comm.ser.is_open:
                print(f"Attempting to switch to serial port: {new_port_name}")
                if self.serial_comm.connect(new_port_name, self.serial_comm.baudrate):
                    self.selected_serial_port = new_port_name
                    print(f"Successfully connected to {self.selected_serial_port}")
                else:
                    QtWidgets.QMessageBox.warning(self, "序列埠連線錯誤",
                                                  f"無法連接至 {new_port_name}。")
                    radio_button.setChecked(False)
                    if self.selected_serial_port:
                        for btn in self.serial_port_radio_group.buttons():
                            if btn.property("port_name") == self.selected_serial_port:
                                btn.setChecked(True)
                                if not (self.serial_comm.ser and self.serial_comm.ser.is_open and self.serial_comm.port == self.selected_serial_port) :
                                    self.serial_comm.connect(self.selected_serial_port, self.serial_comm.baudrate)
                                break
                    else:
                        self.serial_comm.close()


    def closeEvent(self, event):
        if self.opencv_thread:
            self.opencv_thread.stop()
        if self.cap and self.cap.isOpened(): # Check if cap exists and is opened
            self.cap.release()
        if self.serial_comm:
            self.serial_comm.close()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()