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
import platform # <-- 新增：用於偵測作業系統
import serial.tools.list_ports

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
        # ... (保持不變) ...
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
        # ... (保持不變) ...
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
                            # 如果 buffer 中已有內容，加上這次讀到的（不含換行）再返回
                            full_line = self.response_buffer + line
                            self.response_buffer = "" # 清空 buffer
                            return full_line.strip()
                        else:
                            line += char
                    except UnicodeDecodeError:
                        #忽略無法解碼的位元組，或者可以記錄錯誤
                        pass
                else:
                    #短暫休眠，避免CPU空轉
                    time.sleep(0.01)
            # 超時，返回當前已讀取的（可能不完整）
            self.response_buffer += line # 將未完成的行存回 buffer
            if self.response_buffer:
                # print(f"Timeout, partial response in buffer: {self.response_buffer}")
                pass
            return None

        except Exception as e:
            print(f"Error reading serial response: {e}")
            self.response_buffer = "" # 出錯時清空 buffer
            return None
    def close(self):
        if self.ser and self.ser.is_open:
            current_port = self.port # 記錄一下是哪個 port 被關閉
            self.ser.close()
            print(f"Serial connection closed on {current_port if current_port else 'N/A'}.")
        self.ser = None # 確保 ser 在關閉後為 None
        # self.port = None # 可選擇是否在 close 時清除 port 記錄
    @staticmethod
    def get_available_ports():
        """獲取系統上所有可用的序列埠列表"""
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        print(f"Available serial ports: {available_ports}")
        return available_ports

    @staticmethod
    def get_os_specific_default_port():
        """根據作業系統返回一個常見的預設序列埠名稱 (僅為建議)"""
        os_type = platform.system().lower()
        if "windows" in os_type:
            return "COM8" # Windows 上的常見名稱 (與原程式相同)
        elif "linux" in os_type:
            # Linux 上可能是 /dev/ttyUSB0, /dev/ttyACM0, /dev/ttyS0 等
            # 這裡只給一個常見的，實際應從 available_ports 選擇
            return "/dev/ttyUSB0"
        return None


class ColorProfile:
    # c1, c2, c3 分別代表不同色彩空間下的通道1, 通道2, 通道3
    # 例如 HSV 下是 H, S, V; LAB 下是 L, A, B
    def __init__(self, name, c1_min=0, c1_max=179, c2_min=0, c2_max=255, c3_min=0, c3_max=255, color_hex="#FFFFFF"):
        self.name = name
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2_min = c2_min
        self.c2_max = c2_max
        self.c3_min = c3_min
        self.c3_max = c3_max
        self.color_hex = color_hex  # 用於 UI 列表顯示的顏色
        self.active = True  # 是否偵測此顏色

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
    update_frame = pyqtSignal(QImage) #訊號名稱與原程式相同

    def __init__(self, widget_ref, cap_ref): #傳入 widget 的參考和 cap 的參考
        super().__init__()
        self.widget = widget_ref
        self.cap = cap_ref # 使用從 MyWidget 傳入的 VideoCapture 物件
        self.running = True
        self.mouse_press_count = 0 # 原程式的 press_count
        self.mouse_x_coords = [0, 0] # 原程式的 mouse_x
        self.mouse_y_coords = [0, 0] # 原程式的 mouse_y
        self.min_tracking_area = 100 # 可調整的最小追蹤面積

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("無法讀取影像幀 (OpenCVThread)")
                self.msleep(30) # 稍作等待
                continue
            
            # 1. 套用亮度/對比度調整 (調整原始 frame)
            # 從 widget 讀取數值 (假設 MyWidget 中有名為 brightness_val 和 contrast_val 的屬性)
            brightness = self.widget.brightness_val
            contrast_alpha = self.widget.contrast_val / 100.0 # contrast_val 為 0-200 -> alpha 0.0-2.0
            
            try:
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_alpha, beta=brightness)
            except cv2.error as e:
                print(f"OpenCV 亮度/對比度調整錯誤: {e}")
                adjusted_frame = frame.copy() # 錯誤則使用原始幀


            # 2. 處理滑鼠選取顏色 (在原始 frame 上操作，避免受亮度/對比度影響)
            if self.mouse_press_count == 1 and self.widget.current_profile_index >= 0:
                self.process_color_selection(frame) # 使用原始 frame
                self.mouse_press_count = 0 # 處理後重置

            # 3. 處理影像幀 (使用 adjusted_frame)
            processed_display_frame = self.process_image_for_display(adjusted_frame)

            # 4. 轉換圖像並發送信號
            qimg = self.convert_cv_to_qimage(processed_display_frame)
            if qimg: # 確保 qimg 有效
                 self.update_frame.emit(qimg)
            self.msleep(10) # 控制幀率，避免CPU過載

    def process_color_selection(self, original_frame): #接收原始幀
        # 這部分邏輯與原程式的 process_color_selection 類似，但使用 frame 座標
        # 並調用 get_color_range_from_selection
        x1_widget, x2_widget = sorted([self.mouse_x_coords[0], self.mouse_x_coords[1]])
        y1_widget, y2_widget = sorted([self.mouse_y_coords[0], self.mouse_y_coords[1]])

        # 將 widget 座標轉換為 QLabel 內的座標
        label_geom = self.widget.label.geometry() # QLabel 的幾何資訊
        label_pos_in_widget = self.widget.label.mapTo(self.widget, QPoint(0,0)) # QLabel 左上角在 MyWidget 中的座標

        # 計算選取框在 QLabel 內的相對座標
        x1_label = max(0, x1_widget - label_pos_in_widget.x())
        y1_label = max(0, y1_widget - label_pos_in_widget.y())
        x2_label = min(label_geom.width(), x2_widget - label_pos_in_widget.x())
        y2_label = min(label_geom.height(), y2_widget - label_pos_in_widget.y())
        
        if x1_label < x2_label and y1_label < y2_label:
            # 將 QLabel 內的座標按比例縮放到原始影像幀的座標
            frame_h, frame_w = original_frame.shape[:2]
            label_w, label_h = label_geom.width(), label_geom.height()

            if label_w == 0 or label_h == 0: return # 防止除以零

            scale_x = frame_w / label_w
            scale_y = frame_h / label_h

            x1_frame = int(x1_label * scale_x)
            y1_frame = int(y1_label * scale_y)
            x2_frame = int(x2_label * scale_x)
            y2_frame = int(y2_label * scale_y)

            if x1_frame < x2_frame and y1_frame < y2_frame:
                crop_img = original_frame[y1_frame:y2_frame, x1_frame:x2_frame]
                if crop_img.size > 0:
                    # 使用新的方法並傳入當前色彩空間
                    (c1m, c2m, c3m), (c1x, c2x, c3x) = self.get_color_range_from_selection(crop_img, self.widget.current_color_space)
                    if c1m is not None: # 確保成功獲取範圍
                        self.widget.update_profile_channels(c1m, c1x, c2m, c2x, c3m, c3x)


    def get_color_range_from_selection(self, crop_img, color_space):
        # 此方法取代原程式中的全域 getHSVRange
        # 並根據傳入的 color_space 進行處理
        color_range_sensitivity = 15 # 可調整的選取靈敏度 (原為 colorRange = 10)
        
        target_crop = None
        ch_configs = self.widget.get_channel_configs(color_space) # 從 MyWidget 獲取通道配置

        if color_space == 'HSV': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
        elif color_space == 'YCrCb': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'HLS': target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HLS)
        else: target_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV) # 預設 HSV

        if target_crop.size == 0: return (None,None,None), (None,None,None)

        avg_per_channel = np.average(target_crop, axis=(0, 1))
        
        mins = np.zeros(3, dtype=int)
        maxs = np.zeros(3, dtype=int)

        for i in range(3):
            min_val, max_val = ch_configs['ranges'][i]
            mins[i] = np.clip(avg_per_channel[i] - color_range_sensitivity, min_val, max_val).astype(int)
            maxs[i] = np.clip(avg_per_channel[i] + color_range_sensitivity, min_val, max_val).astype(int)
            if mins[i] >= maxs[i]: # 確保 min < max
                if maxs[i] < max_val: maxs[i] = mins[i] + 1
                elif mins[i] > min_val: mins[i] = maxs[i] -1
                else: # 如果在邊界且無法形成有效範圍，則可能需要特殊處理或返回None
                    pass # 保持原樣，可能導致範圍為0

        return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])

    def process_image_for_display(self, adjusted_frame): #取代原 process_frame
        # 1. 色彩空間轉換
        current_cs = self.widget.current_color_space
        if current_cs == 'HSV': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        elif current_cs == 'LAB': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2LAB)
        elif current_cs == 'YCrCb': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2YCrCb)
        elif current_cs == 'HLS': converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HLS)
        else: converted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV) # Default

        frame_height, frame_width = adjusted_frame.shape[:2]
        
        # 根據顯示模式準備輸出幀
        # 'original_with_highlights' 模式下，在 adjusted_frame 的副本上繪圖
        # 'masked' 模式下，最終結果是遮罩後的 adjusted_frame
        output_display_frame = adjusted_frame.copy() 
        
        all_masks_combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
        detected_objects_for_tracking = [] # 用於追蹤決策

        # 2. 遍歷顏色設定檔進行偵測
        for profile in self.widget.color_profiles:
            if not profile.active:
                continue
                
            lower_bound = np.array([profile.c1_min, profile.c2_min, profile.c3_min], dtype=np.uint8)
            upper_bound = np.array([profile.c1_max, profile.c2_max, profile.c3_max], dtype=np.uint8)
            
            individual_mask = cv2.inRange(converted_frame, lower_bound, upper_bound)
            
            # 可選: 形態學操作去噪
            kernel = np.ones((5,5), np.uint8)
            individual_mask = cv2.morphologyEx(individual_mask, cv2.MORPH_OPEN, kernel)
            individual_mask = cv2.morphologyEx(individual_mask, cv2.MORPH_CLOSE, kernel)
            
            all_masks_combined = cv2.bitwise_or(all_masks_combined, individual_mask)
            
            # 為每個 profile 尋找最大輪廓
            contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_contour_for_profile = None
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_tracking_area and area > max_area: # 加入最小面積判斷
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
                    
                    # 如果不是遮罩模式，直接在 output_display_frame 上繪製高亮
                    if self.widget.display_mode == 'original_with_highlights':
                        q_color = QColor(profile.color_hex)
                        bgr_color = (q_color.blue(), q_color.green(), q_color.red())
                        cv2.rectangle(output_display_frame, (x, y), (x + w, y + h), bgr_color, 2)
                        cv2.circle(output_display_frame, (cx, cy), 7, bgr_color, -1)
                        cv2.putText(output_display_frame, profile.name, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        # 3. 追蹤邏輯 (如果啟用)
        if self.widget.tracking_center and detected_objects_for_tracking:
            # 策略：追蹤面積最大的物件
            primary_target = max(detected_objects_for_tracking, key=lambda obj: obj['area'])
            
            cx = primary_target['cx']
            # cy = primary_target['cy'] # 目前未使用
            w_obj = primary_target['w']
            h_obj = primary_target['h']
            
            frame_width = adjusted_frame.shape[1]
            frame_center_x = frame_width // 2
            # 在顯示的幀上畫中心線和追蹤目標指示 (即使是 masked view，也畫在最終的 frame 上)
            cv2.line(output_display_frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 0, 0), 2) # 藍色中心線
            q_color_target = QColor(primary_target['profile_color_hex'])
            bgr_color_target = (q_color_target.blue(), q_color_target.green(), q_color_target.red())
            cv2.putText(output_display_frame, f"Tracking: {primary_target['profile_name']}", (cx - w_obj//2, primary_target['cy'] - h_obj//2 -10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color_target, 2)


            # 序列埠指令邏輯 (與原程式類似)
            dead_zone = 20 # 可調整的中心容差範圍
            command_sent = False
            if cx < frame_center_x - dead_zone:
                # self.widget.serial_comm.send_protocol_command("TRNL", [10]) # 左轉10度 (範例參數)
                self.widget.serial_comm.send_protocol_command("run,15,15,0") # 參數: 負值代表左轉
                command_sent = True
            elif cx > frame_center_x + dead_zone:
                # self.widget.serial_comm.send_protocol_command("TRNR", [10]) # 右轉10度 (範例參數)
                self.widget.serial_comm.send_protocol_command("run,15,-15,0")  # 參數: 正值代表右轉
                command_sent = True
            else: # 在中心區域
                # 可以定義更細緻的前進後退指令和參數
                target_speed_or_steps = 0
                if w_obj > 150 or h_obj > 150:
                    # self.widget.serial_comm.send_protocol_command("WALKB", [15, 1]) # 後退, 步長15, 1步
                    self.widget.serial_comm.send_protocol_command("run,-15,0,0") # 參數: 負值代表後退
                    command_sent = True
                elif w_obj > 100 or h_obj > 100:
                    # self.widget.serial_comm.send_protocol_command("WALKF", [10, 1]) # 慢速前進, 步長10, 1步
                    self.widget.serial_comm.send_protocol_command("run,-10,0,0")  # 參數: 正值代表前進
                    command_sent = True
                else:
                    # self.widget.serial_comm.send_protocol_command("WALKF", [20, 1]) # 正常前進, 步長20, 1步
                    self.widget.serial_comm.send_protocol_command("run,30,0,0")
                    command_sent = True
            
            # res = self.widget.serial_comm.ser.readline().decode('utf-8').strip() # 讀取回應的邏輯，如果需要
            # if res: print(f"Response: {res}")


        # 4. 決定最終顯示的幀
        if self.widget.display_mode == 'masked':
            final_result_frame = cv2.bitwise_and(adjusted_frame, adjusted_frame, mask=all_masks_combined)
            # 如果需要在 masked view 也顯示追蹤線條和文字，可以畫在 final_result_frame 上
            if self.widget.tracking_center and detected_objects_for_tracking:
                 # 重新畫追蹤相關的線條，因為 bitwise_and 會清除它們
                primary_target = max(detected_objects_for_tracking, key=lambda obj: obj['area'])
                cx_trk = primary_target['cx']
                frame_center_x_trk = frame_width // 2
                cv2.line(final_result_frame, (frame_center_x_trk, 0), (frame_center_x_trk, frame_height), (255, 0, 0), 2)
                q_color_target_trk = QColor(primary_target['profile_color_hex'])
                bgr_color_target_trk = (q_color_target_trk.blue(), q_color_target_trk.green(), q_color_target_trk.red())
                cv2.putText(final_result_frame, f"Tracking: {primary_target['profile_name']}", (cx_trk - primary_target['w']//2, primary_target['cy'] - primary_target['h']//2 -10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color_target_trk, 2)


        else: # 'original_with_highlights'
            final_result_frame = output_display_frame
            
        return final_result_frame


    def convert_cv_to_qimage(self, cv_img): # 原程式的 convert_to_qimage
        if cv_img is None: return None
        try:
            target_w = self.widget.label.width()
            target_h = self.widget.label.height()

            if target_w <= 0 or target_h <=0 : # Label 可能尚未完全初始化
                return None

            # 確保 cv_img 不是空的
            if cv_img.size == 0:
                return None

            # 縮放影像以符合 QLabel 大小
            resized_img = cv2.resize(cv_img, (target_w, target_h)) #, interpolation=cv2.INTER_AREA)

            # BGR to RGB
            if len(resized_img.shape) == 3: # Color image
                rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif len(resized_img.shape) == 2: # Grayscale image
                h, w = resized_img.shape
                bytes_per_line = w
                return QImage(resized_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            return None
        except Exception as e:
            print(f"Error in convert_cv_to_qimage: {e}")
            return None


    def stop(self): # 原程式的 stop
        self.running = False
        self.wait()

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.color_profiles = []
        self.current_profile_index = -1
        
        # Feature 1: 色彩空間
        self.current_color_space = 'HSV' # 預設色彩空間
        self.supported_color_spaces = ['HSV', 'LAB', 'YCrCb', 'HLS'] 
        
        # Feature 2: 亮度/對比度
        self.brightness_val = 0  # 亮度調整值 (-100 到 100)
        self.contrast_val = 100  # 對比度調整值 (0 到 200, 代表 0.0 到 2.0)

        # 原程式的 display_mode 和 tracking_center
        self.display_mode = 'masked' # 預設與原程式相同
        self.tracking_center = False
        
        self.serial_comm = SerialCommunication(baudrate=1000000) # 先不指定 port
        self.available_serial_ports = SerialCommunication.get_available_ports()
        self.selected_serial_port = None # 用於追蹤目前UI上選擇或自動連接的port

        if self.available_serial_ports:
            # 嘗試自動連接：優先使用OS特定的預設埠名，如果存在的話
            os_default_port = SerialCommunication.get_os_specific_default_port()
            port_to_try = None

            if os_default_port and os_default_port in self.available_serial_ports:
                port_to_try = os_default_port
            else: # 否則使用列表中的第一個
                port_to_try = self.available_serial_ports[0]
            
            if self.serial_comm.connect(port_to_try, self.serial_comm.baudrate):
                self.selected_serial_port = port_to_try
                print(f"Auto-connected to serial port: {self.selected_serial_port}")
            else:
                QtWidgets.QMessageBox.warning(self, "序列埠錯誤", f"無法自動連接至 {port_to_try}。請手動選擇。")
        else:
            QtWidgets.QMessageBox.warning(self, "序列埠錯誤", "未偵測到任何序列埠！")
            print("No serial ports detected.")
        
        # 初始化攝影機
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "攝影機錯誤", "無法開啟攝影機！")
            # 在這裡可以選擇退出或禁用相關功能
            # sys.exit() # 或者更優雅地處理
            self.cap = None # 標記攝影機不可用

        # QLabel 用於顯示影像 (原程式的 self.label)
        self.label = QtWidgets.QLabel(self) # 名稱與原程式相同
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;") # 預設背景
        
        # 建立 OpenCVThread
        if self.cap: # 僅當攝影機成功開啟時才建立執行緒
            self.opencv_thread = OpenCVThread(self, self.cap)
        else:
            self.opencv_thread = None

        self.slider_labels = {} # 用於存放滑桿旁邊的 QLabel 物件
        self.slider_value_labels = {} # 用於存放顯示滑桿數值的 QLabel

        self.initUI() # 初始化 UI 元件

        # 新增一個預設設定檔並選取它
        self.add_new_profile("預設") # 原程式的 "預設"
        if self.color_profiles:
            self.select_profile_by_index(0)
        
        if self.opencv_thread:
            self.opencv_thread.update_frame.connect(self.update_label)
            self.opencv_thread.start()
        else:
            self.label.setText("攝影機未連接或初始化失敗")


    def get_channel_configs(self, color_space_name=None):
        """提供指定色彩空間的通道名稱和範圍"""
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
        else: # 預設回 HSV
            return {'names': ['H', 'S', 'V'], 'ranges': [(0, 179), (0, 255), (0, 255)]}


    def initUI(self):
        self.setWindowTitle('WeAreMinions') # 更新標題
        self.resize(1200, 750) # 稍微調整視窗大小

        main_layout = QtWidgets.QHBoxLayout(self)
        
        # --- 左側控制面板 ---
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(10) # 調整間距

        # 顏色設定檔管理 (與原程式類似)
        profile_group = QtWidgets.QGroupBox("Color Profiles")
        profile_layout = QtWidgets.QVBoxLayout()
        self.profile_list = QtWidgets.QListWidget() # 原 self.profile_list
        self.profile_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.profile_list.itemClicked.connect(self.select_profile_from_list_item) # 原 self.select_profile
        profile_layout.addWidget(self.profile_list)
        
        profile_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self.add_profile_dialog)
        profile_btn_layout.addWidget(self.btn_add)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_current_profile)
        profile_btn_layout.addWidget(self.btn_remove)
        self.btn_toggle_active = QtWidgets.QPushButton("Toggle Active") #原 self.btn_toggle
        self.btn_toggle_active.clicked.connect(self.toggle_current_profile_active) #原 self.toggle_current_profile
        profile_btn_layout.addWidget(self.btn_toggle_active)
        profile_layout.addLayout(profile_btn_layout)
        profile_group.setLayout(profile_layout)
        left_panel.addWidget(profile_group)


        serial_port_group = QtWidgets.QGroupBox("Serial Port Selection")
        serial_port_layout = QtWidgets.QVBoxLayout()
        self.serial_port_radio_group = QtWidgets.QButtonGroup(self) # 用於互斥選擇

        if not self.available_serial_ports:
            no_ports_label = QtWidgets.QLabel("未找到序列埠 (No serial ports found)")
            serial_port_layout.addWidget(no_ports_label)
        else:
            for i, port_name in enumerate(self.available_serial_ports):
                rb = QtWidgets.QRadioButton(port_name)
                rb.setProperty("port_name", port_name) # 將埠名存儲在radio button中
                if port_name == self.selected_serial_port: # 如果是自動連接的埠，則預設選中
                    rb.setChecked(True)
                rb.toggled.connect(self.on_serial_port_changed) # 連接訊號
                serial_port_layout.addWidget(rb)
                self.serial_port_radio_group.addButton(rb, i)
        
        serial_port_group.setLayout(serial_port_layout)
        left_panel.addWidget(serial_port_group)


        # 色彩空間切換控制
        cs_group = QtWidgets.QGroupBox("Color Space")
        cs_layout = QtWidgets.QVBoxLayout()
        self.current_cs_label = QtWidgets.QLabel(f"Current: {self.current_color_space}")
        cs_layout.addWidget(self.current_cs_label)
        btn_change_cs = QtWidgets.QPushButton("Change Color Space...")
        btn_change_cs.clicked.connect(self.show_color_space_dialog)
        cs_layout.addWidget(btn_change_cs)
        cs_group.setLayout(cs_layout)
        left_panel.addWidget(cs_group)

        # 通道滑桿 (原 HSV 滑桿，現為通用通道)
        slider_group = QtWidgets.QGroupBox("Channel Controls")
        slider_main_layout = QtWidgets.QVBoxLayout() #一個垂直佈局容納三組通道

        # 使用 self.slider_labels 和 self.slider_value_labels 字典來儲存
        # 原來的 self.slider_H_min 等變數將對應到 c1, c2, c3
        self.slider_c1_min, self.slider_labels['c1_min'], self.slider_value_labels['c1_min'] = self.create_slider_pair("C1", 0, 179, slider_main_layout)
        self.slider_c1_max = self.slider_c1_min.property("companion_slider") # 獲取配對的 max 滑桿

        self.slider_c2_min, self.slider_labels['c2_min'], self.slider_value_labels['c2_min'] = self.create_slider_pair("C2", 0, 255, slider_main_layout)
        self.slider_c2_max = self.slider_c2_min.property("companion_slider")
        
        self.slider_c3_min, self.slider_labels['c3_min'], self.slider_value_labels['c3_min'] = self.create_slider_pair("C3", 0, 255, slider_main_layout)
        self.slider_c3_max = self.slider_c3_min.property("companion_slider")

        # 連接訊號 (只需連接 min 滑桿，max 滑桿由 min 滑桿的 valueChanged 間接觸發)
        self.slider_c1_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c2_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c3_min.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c1_max.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c2_max.valueChanged.connect(self.update_current_profile_from_sliders)
        self.slider_c3_max.valueChanged.connect(self.update_current_profile_from_sliders)


        self.update_sliders_for_color_space() # 根據預設色彩空間初始化滑桿標籤和範圍

        slider_group.setLayout(slider_main_layout)
        left_panel.addWidget(slider_group)

        # 影像調整 (亮度/對比度)
        img_adj_group = QtWidgets.QGroupBox("Image Adjustments")
        img_adj_layout = QtWidgets.QVBoxLayout()
        
        # 亮度
        brightness_label_widget = QtWidgets.QLabel("Brightness: 0")
        img_adj_layout.addWidget(brightness_label_widget)
        self.slider_brightness = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(self.brightness_val)
        self.slider_brightness.valueChanged.connect(lambda v, lbl=brightness_label_widget: (
            self.brightness_changed(v), lbl.setText(f"Brightness: {v}")
        ))
        img_adj_layout.addWidget(self.slider_brightness)

        # 對比度
        contrast_label_widget = QtWidgets.QLabel(f"Contrast: {self.contrast_val/100.0:.2f}x")
        img_adj_layout.addWidget(contrast_label_widget)
        self.slider_contrast = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 200) # 0-200 代表 0.0x - 2.0x
        self.slider_contrast.setValue(self.contrast_val)
        self.slider_contrast.valueChanged.connect(lambda v, lbl=contrast_label_widget: (
            self.contrast_changed(v), lbl.setText(f"Contrast: {v/100.0:.2f}x")
        ))
        img_adj_layout.addWidget(self.slider_contrast)
        img_adj_group.setLayout(img_adj_layout)
        left_panel.addWidget(img_adj_group)


        # 選取說明 (與原程式相同)
        instructions = QtWidgets.QLabel("Click to select area")
        instructions.setWordWrap(True)
        left_panel.addWidget(instructions)
        
        # 檔案操作 (與原程式相同)
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
        
        left_panel.addStretch(1) # 使控制項靠上

        main_layout.addLayout(left_panel, 1)  # 左側面板佔用 1/3 寬度
        
        # --- 右側影像顯示面板 ---
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(self.label, stretch=1) # 影像標籤佔用主要空間
        
        # 顯示與追蹤選項 (合併原程式的 display_group 和 tracking_group)
        options_group = QtWidgets.QGroupBox("Display & Tracking")
        options_layout = QtWidgets.QHBoxLayout()
        
        self.radio_masked = QtWidgets.QRadioButton("Masked")
        self.radio_original = QtWidgets.QRadioButton("Original+Highlights") #原 original
        
        if self.display_mode == 'masked':
            self.radio_masked.setChecked(True)
        else:
            self.radio_original.setChecked(True)
            
        self.radio_masked.clicked.connect(self.change_display_mode)
        self.radio_original.clicked.connect(self.change_display_mode)
        options_layout.addWidget(self.radio_masked)
        options_layout.addWidget(self.radio_original)
        
        self.btn_track_center = QtWidgets.QPushButton("Center Tracking") #原 btn_track_center
        self.btn_track_center.setCheckable(True) # 設為可選中狀態
        self.btn_track_center.setChecked(self.tracking_center)
        self.btn_track_center.clicked.connect(self.toggle_center_tracking)
        options_layout.addWidget(self.btn_track_center)
        
        options_group.setLayout(options_layout)
        right_panel.addWidget(options_group)
        
        main_layout.addLayout(right_panel, 2) # 右側面板佔用 2/3 寬度

    def create_slider_pair(self, base_name, min_val_default, max_val_default, parent_layout):
        """ 建立一對 Min/Max 滑桿和對應的標籤 """
        # Min 滑桿部分
        min_label_text = f"{base_name} Min:"
        min_slider_container = QtWidgets.QHBoxLayout()
        
        actual_min_label = QtWidgets.QLabel(min_label_text) # 實際的通道標籤
        self.slider_labels[f'{base_name.lower()}_min_prefix'] = actual_min_label # 儲存前綴標籤
        min_slider_container.addWidget(actual_min_label)

        min_slider = QtWidgets.QSlider(Qt.Horizontal)
        min_slider.setRange(min_val_default, max_val_default) # 預設範圍
        min_slider_container.addWidget(min_slider)
        
        min_value_label = QtWidgets.QLabel(str(min_slider.value())) # 顯示數值的標籤
        min_slider.valueChanged.connect(lambda v, lbl=min_value_label, s=min_slider: (
            lbl.setText(str(v)),
            self.update_current_profile_from_sliders() # 值改變時立即更新
        ))
        min_slider_container.addWidget(min_value_label)
        self.slider_value_labels[f'{base_name.lower()}_min'] = min_value_label #儲存數值標籤

        parent_layout.addLayout(min_slider_container)

        # Max 滑桿部分
        max_label_text = f"{base_name} Max:"
        max_slider_container = QtWidgets.QHBoxLayout()

        actual_max_label = QtWidgets.QLabel(max_label_text)
        self.slider_labels[f'{base_name.lower()}_max_prefix'] = actual_max_label
        max_slider_container.addWidget(actual_max_label)

        max_slider = QtWidgets.QSlider(Qt.Horizontal)
        max_slider.setRange(min_val_default, max_val_default)
        max_slider.setValue(max_val_default) # Max 通常預設為最大值
        max_slider_container.addWidget(max_slider)

        max_value_label = QtWidgets.QLabel(str(max_slider.value()))
        max_slider.valueChanged.connect(lambda v, lbl=max_value_label, s=max_slider: (
            lbl.setText(str(v)),
            self.update_current_profile_from_sliders() # 值改變時立即更新
        ))
        max_slider_container.addWidget(max_value_label)
        self.slider_value_labels[f'{base_name.lower()}_max'] = max_value_label

        parent_layout.addLayout(max_slider_container)

        # 互相設定為對方的 companion，方便後續同時操作 (例如更新範圍)
        min_slider.setProperty("companion_slider", max_slider)
        max_slider.setProperty("companion_slider", min_slider)
        
        return min_slider, actual_min_label, min_value_label # 回傳 Min 滑桿和其標籤


    def brightness_changed(self, value):
        self.brightness_val = value

    def contrast_changed(self, value):
        self.contrast_val = value


    def update_sliders_for_color_space(self):
        """ 根據當前的 self.current_color_space 更新滑桿的標籤和範圍 """
        configs = self.get_channel_configs() # 獲取當前色彩空間的配置
        ch_names = configs['names']
        ch_ranges = configs['ranges']

        sliders = {
            'c1': (self.slider_c1_min, self.slider_c1_max),
            'c2': (self.slider_c2_min, self.slider_c2_max),
            'c3': (self.slider_c3_min, self.slider_c3_max)
        }

        for i, key_base in enumerate(['c1', 'c2', 'c3']):
            min_slider, max_slider = sliders[key_base]
            
            # 更新標籤文字
            self.slider_labels[f'{key_base}_min_prefix'].setText(f"{ch_names[i]} Min:")
            self.slider_labels[f'{key_base}_max_prefix'].setText(f"{ch_names[i]} Max:")
            
            # 更新滑桿範圍
            min_slider.setRange(ch_ranges[i][0], ch_ranges[i][1])
            max_slider.setRange(ch_ranges[i][0], ch_ranges[i][1])

        # 如果當前有選定的設定檔，重新將其數值載入到滑桿上
        # 這會確保滑桿顯示的值在新的範圍內被正確限制
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
            self.cs_radio_button_group.addButton(rb, i) # 將索引與按鈕關聯

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        
        def accept_selection():
            selected_id = self.cs_radio_button_group.checkedId()
            if selected_id != -1: # -1 表示沒有按鈕被選中
                new_space = self.supported_color_spaces[selected_id]
                if new_space != self.current_color_space:
                    self.current_color_space = new_space
                    self.current_cs_label.setText(f"當前: {self.current_color_space}")
                    self.update_sliders_for_color_space() # 更新滑桿
            dialog.accept()

        button_box.accepted.connect(accept_selection)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.exec_()


    def add_new_profile(self, name): # 與原程式類似，但使用 c1, c2, c3
        # 產生隨機顏色 (與原程式相同)
        q_color = QColor(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color_hex = q_color.name()
        
        # 獲取當前色彩空間的預設通道範圍
        configs = self.get_channel_configs()
        
        new_profile = ColorProfile(
            name=name,
            c1_min=configs['ranges'][0][0], c1_max=configs['ranges'][0][1], # 使用當前空間的預設範圍
            c2_min=configs['ranges'][1][0], c2_max=configs['ranges'][1][1],
            c3_min=configs['ranges'][2][0], c3_max=configs['ranges'][2][1],
            color_hex=color_hex
        )
        self.color_profiles.append(new_profile)
        self.update_profile_list_display() # 原 self.update_profile_list
        self.select_profile_by_index(len(self.color_profiles) - 1)
        return len(self.color_profiles) - 1


    def add_profile_dialog(self): # 與原程式類似
        name, ok = QtWidgets.QInputDialog.getText(self, "新增顏色設定檔", "輸入設定檔名稱:")
        if ok and name:
            self.add_new_profile(name)


    def remove_current_profile(self): # 與原程式類似
        if self.current_profile_index >= 0: # 確保有選定的設定檔
            if len(self.color_profiles) <= 1 and self.current_profile_index == 0 : # 如果只剩一個，不允許刪除
                 QtWidgets.QMessageBox.warning(self, "警告", "至少需要保留一個顏色設定檔。")
                 return

            del self.color_profiles[self.current_profile_index]
            self.update_profile_list_display()
            # 更新選取的索引，避免超出範圍
            if self.color_profiles: # 如果列表不為空
                new_index = max(0, min(self.current_profile_index, len(self.color_profiles) - 1))
                self.select_profile_by_index(new_index)
            else: # 如果列表為空
                self.current_profile_index = -1
                self.clear_sliders() # 清空滑桿值
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "沒有選定的設定檔可以移除。")


    def toggle_current_profile_active(self): # 原 toggle_current_profile
        if self.current_profile_index >= 0 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            profile.active = not profile.active
            self.update_profile_list_display()


    def update_profile_list_display(self): # 原 update_profile_list
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

    def select_profile_from_list_item(self, item): # 原 select_profile
        index = item.data(Qt.UserRole)
        self.select_profile_by_index(index)

    def select_profile_by_index(self, index): # 原 select_profile_by_index
        if 0 <= index < len(self.color_profiles):
            self.current_profile_index = index
            profile = self.color_profiles[index]
            self.load_profile_values_to_sliders(profile)
            self.update_profile_list_display() # 更新列表高亮
        elif not self.color_profiles: # 如果列表空了
            self.current_profile_index = -1
            self.clear_sliders()
            self.update_profile_list_display()


    def load_profile_values_to_sliders(self, profile):
        """ 將指定設定檔的數值載入到對應的滑桿上 """
        sliders_map = {
            'c1': (self.slider_c1_min, self.slider_c1_max, profile.c1_min, profile.c1_max),
            'c2': (self.slider_c2_min, self.slider_c2_max, profile.c2_min, profile.c2_max),
            'c3': (self.slider_c3_min, self.slider_c3_max, profile.c3_min, profile.c3_max)
        }
        configs = self.get_channel_configs() # 獲取當前色彩空間的範圍

        for i, key in enumerate(['c1', 'c2', 'c3']):
            slider_min, slider_max, val_min, val_max = sliders_map[key]
            ch_range_min, ch_range_max = configs['ranges'][i]

            slider_min.blockSignals(True)
            slider_max.blockSignals(True)

            slider_min.setValue(np.clip(val_min, ch_range_min, ch_range_max))
            slider_max.setValue(np.clip(val_max, ch_range_min, ch_range_max))
            
            # 更新數值標籤
            self.slider_value_labels[f'{key}_min'].setText(str(slider_min.value()))
            self.slider_value_labels[f'{key}_max'].setText(str(slider_max.value()))

            slider_min.blockSignals(False)
            slider_max.blockSignals(False)
            
    def clear_sliders(self):
        """ 清空滑桿值或設為預設值 """
        sliders_all = [
            self.slider_c1_min, self.slider_c1_max,
            self.slider_c2_min, self.slider_c2_max,
            self.slider_c3_min, self.slider_c3_max
        ]
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


    def update_current_profile_from_sliders(self): # 原 updateCurrentProfile，現在更通用
        if self.current_profile_index != -1 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            
            # 從滑桿讀取值並更新 profile
            # 確保讀取的值在滑桿的當前範圍內 (理論上滑桿本身會限制)
            profile.c1_min = self.slider_c1_min.value()
            profile.c1_max = self.slider_c1_max.value()
            profile.c2_min = self.slider_c2_min.value()
            profile.c2_max = self.slider_c2_max.value()
            profile.c3_min = self.slider_c3_min.value()
            profile.c3_max = self.slider_c3_max.value()

            # 確保 min <= max
            if profile.c1_min > profile.c1_max: profile.c1_min = profile.c1_max
            if profile.c2_min > profile.c2_max: profile.c2_min = profile.c2_max
            if profile.c3_min > profile.c3_max: profile.c3_min = profile.c3_max


    def update_profile_channels(self, c1m, c1x, c2m, c2x, c3m, c3x): # 原 update_profile_hsv
        if self.current_profile_index != -1 and self.color_profiles:
            profile = self.color_profiles[self.current_profile_index]
            profile.c1_min, profile.c1_max = c1m, c1x
            profile.c2_min, profile.c2_max = c2m, c2x
            profile.c3_min, profile.c3_max = c3m, c3x
            self.load_profile_values_to_sliders(profile) # 將新數值更新到滑桿
            self.update_profile_list_display() # 如果需要在列表上反映某些狀態


    def mousePressEvent(self, event): # 與原程式相同
        if self.opencv_thread and event.buttons() == Qt.LeftButton and self.label.geometry().contains(event.pos()):
            # 將 MyWidget 內的座標傳給 OpenCVThread
            self.opencv_thread.mouse_x_coords[0] = event.pos().x()
            self.opencv_thread.mouse_y_coords[0] = event.pos().y()


    def mouseReleaseEvent(self, event): # 與原程式相同
        if self.opencv_thread and event.button() == Qt.LeftButton and self.label.geometry().contains(event.pos()):
            self.opencv_thread.mouse_x_coords[1] = event.pos().x()
            self.opencv_thread.mouse_y_coords[1] = event.pos().y()
            if self.current_profile_index != -1 : # 確保有選定的 profile
                 self.opencv_thread.mouse_press_count = 1
            else:
                 QtWidgets.QMessageBox.information(self, "提示", "請先新增或選取一個顏色設定檔。")


    def saveData(self): # 與原程式類似，但儲存 c1, c2, c3
        filename = self.input_field.text()
        if not filename:
            QtWidgets.QMessageBox.warning(self, "警告", "請輸入檔案名稱")
            return
            
        profiles_data = [profile.to_dict() for profile in self.color_profiles]
        # 也儲存當前的色彩空間，以便載入時恢復
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
    
    def loadData(self): # 與原程式類似，但載入 c1, c2, c3 並恢復色彩空間等
        # 原程式的 loadData 邏輯非常複雜，這裡簡化為標準的檔案選擇對話框
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "載入設定檔", "",
                                                            "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                backup_profiles = self.color_profiles.copy() # 備份
                
                loaded_profiles_data = data.get("profiles", [])
                self.color_profiles = [ColorProfile.from_dict(pd) for pd in loaded_profiles_data]

                # 恢復色彩空間
                loaded_cs = data.get("current_color_space", "HSV")
                if loaded_cs in self.supported_color_spaces:
                    self.current_color_space = loaded_cs
                else:
                    self.current_color_space = "HSV" # 預設
                self.current_cs_label.setText(f"當前: {self.current_color_space}")
                self.update_sliders_for_color_space() # 更新滑桿以匹配新的色彩空間

                # 恢復亮度/對比度
                self.brightness_val = data.get("brightness", 0)
                self.contrast_val = data.get("contrast", 100)
                self.slider_brightness.setValue(self.brightness_val)
                self.slider_contrast.setValue(self.contrast_val)


                self.update_profile_list_display()
                if self.color_profiles:
                    self.select_profile_by_index(0) # 預設選取第一個
                else: # 如果沒有載入任何設定檔
                    self.color_profiles = backup_profiles # 恢復備份
                    QtWidgets.QMessageBox.information(self, "通知", "檔案中未找到有效的設定檔，或設定檔為空。")
                
                self.input_field.setText(os.path.splitext(os.path.basename(fileName))[0])
                QtWidgets.QMessageBox.information(self, "成功", f"已從 {fileName} 載入設定")

            except json.JSONDecodeError:
                QtWidgets.QMessageBox.critical(self, "錯誤", "檔案格式錯誤，無法解析 JSON。")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "錯誤", f"載入失敗: {str(e)}")


    def toggle_center_tracking(self): # 與原程式相同
        self.tracking_center = not self.tracking_center # 直接修改屬性值
        self.btn_track_center.setChecked(self.tracking_center) # 更新按鈕狀態
        self.btn_track_center.setText("停止追蹤 (Stop)" if self.tracking_center else "中心追蹤 (Center Tracking)")


    def change_display_mode(self): # 與原程式相同
        self.display_mode = 'masked' if self.radio_masked.isChecked() else 'original_with_highlights'


    def update_label(self, qimg): # 與原程式相同
        if qimg and not qimg.isNull():
            self.label.setPixmap(QPixmap.fromImage(qimg))
        # else:
            # self.label.setText("無影像訊號") # 如果 qimg 為空或無效


    def on_serial_port_changed(self): # <-- 新增：處理序列埠選擇變更的函數
        """當使用者選擇不同的序列埠時呼叫此函數"""
        radio_button = self.sender() # 獲取觸發信號的 QRadioButton
        if radio_button and radio_button.isChecked():
            new_port_name = radio_button.property("port_name")
            if new_port_name != self.selected_serial_port or not self.serial_comm.ser or not self.serial_comm.ser.is_open:
                print(f"Attempting to switch to serial port: {new_port_name}")
                # 使用之前存儲的 baudrate
                if self.serial_comm.connect(new_port_name, self.serial_comm.baudrate):
                    self.selected_serial_port = new_port_name
                    print(f"Successfully connected to {self.selected_serial_port}")
                else:
                    QtWidgets.QMessageBox.warning(self, "序列埠連線錯誤",
                                                  f"無法連接至 {new_port_name}。")
                    # 連接失敗，嘗試恢復到之前的選擇 (如果有的話)
                    # 或者取消選中當前失敗的 radio button
                    radio_button.setChecked(False) # 這可能會再次觸發 toggled，但因為 isChecked 為 false，通常不會進入邏輯
                    if self.selected_serial_port: # 如果之前有成功連接的 port
                        for btn in self.serial_port_radio_group.buttons():
                            if btn.property("port_name") == self.selected_serial_port:
                                btn.setChecked(True) # 重新選中上一個成功的
                                # 再次嘗試連接舊的port，確保狀態一致
                                if not (self.serial_comm.ser and self.serial_comm.ser.is_open and self.serial_comm.port == self.selected_serial_port) :
                                    self.serial_comm.connect(self.selected_serial_port, self.serial_comm.baudrate)
                                break
                    else: # 如果之前沒有成功連接的port (例如啟動時就失敗了)
                        self.serial_comm.close() # 確保 serial_comm 內部狀態是關閉的
                        # print("No previously selected port to revert to.")


    def closeEvent(self, event):
        if self.opencv_thread:
            self.opencv_thread.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.serial_comm: # <-- 修改：確保 serial_comm 物件存在
            self.serial_comm.close()
        event.accept()

# 全域的 getHSVRange 已被 OpenCVThread 中的 get_color_range_from_selection 取代
# ocv = True # 此變數在原程式中未被使用

def main():
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()