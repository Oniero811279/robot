# MarathonCompetition_ColorOnly.py

import cv2
import numpy as np
import time
import json
import os

try:
    from HSV import SerialCommunication, ColorProfile
except ImportError:
    print("錯誤：無法從 HSV.py 匯入 SerialCommunication 或 ColorProfile。")
    exit()

# --- 馬拉松競賽狀態 ---
STATE_MARATHON_IDLE = 0
STATE_MARATHON_FOLLOWING_LINE = 1
STATE_MARATHON_LINE_LOST = 2 # 線條丟失狀態
STATE_MARATHON_COMPLETED = 3 # 假設比賽完成 (例如，手動停止或計時結束)

# --- 競賽參數與設定 ---
CAMERA_INDEX = 0
PROFILE_FILE_NAME_MARATHON = "marathon.json"
SERIAL_PORT = None
BAUDRATE = 1000000

# 影像處理參數
MIN_LINE_CONTOUR_AREA = 300       # 賽道線條最小輪廓面積 (需校準)
# 線條中心偏移容許範圍 (相對於畫面寬度)
# 例如，畫面寬度640，10% 就是64像素。這裡用比例。
LINE_CENTER_DEADZONE_RATIO = 0.08 # 畫面中心左右各 8% 的區域視為直線，可調整

# 移動控制參數 (需校準)
LINE_FOLLOW_SPEED = 25      # 直行速度
TURN_SPEED_LINE_FOLLOW = 15 # 修正轉向速度/參數
LOST_LINE_SLOW_SPEED = 10   # 丟失線條時的慢速前進速度

MOVE_COMMAND = "run,35,0,0"
TURN_RIGHT_COMMAND = "run,35,10,0"
TURN_LEFT_COMMAND = "run,35,-10,0"

# 預期顏色設定檔名稱
PROFILE_NAME_TRACK_LINE = "TrackLine"

class MarathonRobotColorOnly:
    def __init__(self, serial_comm, color_profiles_dict):
        self.ser = serial_comm
        self.profiles = color_profiles_dict
        self.current_state = STATE_MARATHON_IDLE
        self.marathon_start_time = 0
        self.last_line_seen_time = 0

        self.cap = None
        self.frame_height = 0
        self.frame_width = 0
        self.frame_center_x = 0

    def initialize_camera(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"錯誤：無法開啟攝影機索引 {camera_index}")
            return False
        ret, frame = self.cap.read()
        if not ret:
            print("錯誤：無法從攝影機讀取影像幀")
            self.cap.release(); return False
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_center_x = self.frame_width // 2
        # 計算偏移容許的像素值
        self.line_center_deadzone_pixels = int(self.frame_width * LINE_CENTER_DEADZONE_RATIO)
        print(f"攝影機初始化成功：解析度 {self.frame_width}x{self.frame_height}")
        print(f"線條中心偏移容許範圍：+/- {self.line_center_deadzone_pixels} 像素 (畫面中心左右各 {LINE_CENTER_DEADZONE_RATIO*100:.1f}%)")
        return True

    def _get_profile_bounds(self, profile_name):
        if profile_name not in self.profiles or not self.profiles[profile_name].active:
            return None, None
        profile = self.profiles[profile_name]
        try:
            lower = np.array([profile.c1_min, profile.c2_min, profile.c3_min])
            upper = np.array([profile.c1_max, profile.c2_max, profile.c3_max])
            return lower, upper
        except AttributeError:
            print(f"錯誤：設定檔 {profile_name} 缺少 c1_min 等屬性。")
            return None, None

    def _find_line_center_and_control(self, frame, profile_name_line):
        """
        偵測賽道線條，計算其中心，並根據偏移決定控制指令。
        返回: (line_center_x, error_x, display_frame)
               line_center_x: 線條中心的x座標，若未偵測到則為-1
               error_x: 線條中心與畫面中心的偏差
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_line, upper_line = self._get_profile_bounds(profile_name_line)
        display_frame = frame.copy() # 用於繪製

        if lower_line is None:
            return -1, 0, display_frame # 若設定檔有問題，不進行處理

        mask_line = cv2.inRange(hsv_frame, lower_line, upper_line)
        kernel = np.ones((5, 5), np.uint8)
        mask_line_opened = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, kernel)
        mask_line_closed = cv2.morphologyEx(mask_line_opened, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_line_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_center_x = -1
        error_x = 0

        if contours:
            # 選擇最大面積的輪廓作為賽道線
            line_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(line_contour) > MIN_LINE_CONTOUR_AREA:
                M = cv2.moments(line_contour)
                if M["m00"] != 0:
                    # 計算線條的質心 X 座標
                    line_center_x = int(M["m10"] / M["m00"])
                    line_center_y = int(M["m01"] / M["m00"]) # Y 座標也計算出來用於繪圖

                    # 繪製偵測到的線條和中心點
                    cv2.drawContours(display_frame, [line_contour], -1, (0, 255, 0), 3)
                    cv2.circle(display_frame, (line_center_x, line_center_y), 7, (255, 0, 0), -1)
                    cv2.putText(display_frame, f"Line @ ({line_center_x})", (line_center_x - 50, line_center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # 計算線條中心與畫面中心的偏差
                    error_x = line_center_x - self.frame_center_x
                    self.last_line_seen_time = time.time() # 更新最後看到線條的時間
                    return line_center_x, error_x, display_frame

        # 如果沒有偵測到足夠大的線條輪廓
        return -1, 0, display_frame


    def run_marathon_logic(self):
        if not self.cap or not self.cap.isOpened():
            print("錯誤：攝影機未初始化。")
            return

        print("\n--- HuroCup 馬拉松 (僅顏色偵測直線) ---")
        input("請將機器人放置於起跑線，準備好後按 Enter 開始...")

        self.current_state = STATE_MARATHON_FOLLOWING_LINE
        self.marathon_start_time = time.time()
        self.last_line_seen_time = time.time() # 初始化
        print(f"[{time.strftime('%H:%M:%S')}] 馬拉松開始！狀態：沿線行走")

        last_command_time = time.time()
        COMMAND_INTERVAL = 0.1 # 指令發送間隔 (秒)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("警告：無法讀取影像幀。")
                time.sleep(0.1)
                continue

            # 檢查總比賽時間 (1 小時)
            if (time.time() - self.marathon_start_time) > 4000:
                print(f"[{time.strftime('%H:%M:%S')}] 比賽時間到 (1小時)！")
                self.current_state = STATE_MARATHON_COMPLETED
                self.ser.send_command(MOVE_COMMAND) # 停止機器人
                break

            current_time = time.time()
            send_new_command = (current_time - last_command_time) > COMMAND_INTERVAL
            
            display_frame = frame.copy() # 預設顯示原始影像

            if self.current_state == STATE_MARATHON_FOLLOWING_LINE:
                line_cx, error_x, processed_frame = self._find_line_center_and_control(frame, PROFILE_NAME_TRACK_LINE)
                display_frame = processed_frame # 更新顯示的影像

                if line_cx != -1: # 如果偵測到線條
                    if send_new_command:
                        # 根據 error_x 控制機器人
                        if abs(error_x) <= self.line_center_deadzone_pixels:
                            # 線條在中心容許範圍內，直行
                            self.ser.send_command(MOVE_COMMAND)
                            print(f"  LINE_FOLLOW: 直行 (Error: {error_x}) Speed: 35")
                        elif error_x > self.line_center_deadzone_pixels:
                            # 線條偏右，機器人需向右轉 (假設 TURN 正值為右轉)
                            self.ser.send_command(TURN_RIGHT_COMMAND)
                            print(f"  LINE_FOLLOW: 線條偏右，向右修正 (Error: {error_x}) Turn: 10")
                        else: # error_x < -self.line_center_deadzone_pixels
                            # 線條偏左，機器人需向左轉 (假設 TURN 負值為左轉)
                            self.ser.send_command(TURN_LEFT_COMMAND)
                            print(f"  LINE_FOLLOW: 線條偏左，向左修正 (Error: {error_x}) Turn: -10")
                        last_command_time = current_time
                else: # 未偵測到線條
                    # 檢查多久沒看到線了
                    if (current_time - self.last_line_seen_time) > 3.0: # 例如超過3秒沒看到線
                        print(f"[{time.strftime('%H:%M:%S')}] 線條丟失超過3秒！進入丟失狀態。")
                        self.current_state = STATE_MARATHON_LINE_LOST
                        self.ser.send_command(MOVE_COMMAND, [0]) # 停止或進入特定搜索行為
                    elif send_new_command: # 短暫丟失，可以嘗試慢速直行一小段
                        self.ser.send_command(MOVE_COMMAND, [LOST_LINE_SLOW_SPEED])
                        print(f"  LINE_FOLLOW: 短暫丟失線條，慢速前進 Speed: {LOST_LINE_SLOW_SPEED}")
                        last_command_time = current_time

            elif self.current_state == STATE_MARATHON_LINE_LOST:
                # 線條丟失後的處理邏輯
                # 根據規則，馬拉松有斷裂帶，但這裡沒有箭頭偵測了。
                # 所以，如果線條丟失，機器人基本上無法知道如何繼續。
                # 簡單的處理是停止，或者盲目慢速前進一段時間後停止。
                # 這裡我們讓它在發出警告後，過一段時間就認為比賽無法繼續。
                print(f"[{time.strftime('%H:%M:%S')}] 處於線條丟失狀態。機器人已停止。")
                # 可以加入一個超時，例如10秒後若仍找不到線，則結束。
                if (current_time - self.last_line_seen_time) > 30.0: # 丟失超過15秒
                    print(f"[{time.strftime('%H:%M:%S')}] 線條長時間丟失，比賽中止。")
                    self.current_state = STATE_MARATHON_COMPLETED # 或一個特定的失敗狀態
                    self.ser.send_command(MOVE_COMMAND, [0])
                    break
                # 可以嘗試左右小幅度擺動或旋轉尋找線條，但這裡簡化為停止。
                if send_new_command:
                    self.ser.send_command(MOVE_COMMAND, [0]) # 保持停止
                    last_command_time = current_time


            elif self.current_state == STATE_MARATHON_COMPLETED:
                print(f"[{time.strftime('%H:%M:%S')}] 馬拉松競賽已完成/中止。")
                self.ser.send_command(MOVE_COMMAND, [0])
                break

            # 繪製畫面中心線以供參考
            cv2.line(display_frame, (self.frame_center_x, 0), (self.frame_center_x, self.frame_height), (128, 128, 128), 1)
            # 繪製偏移容許範圍的邊界
            cv2.line(display_frame, (self.frame_center_x - self.line_center_deadzone_pixels, 0),
                       (self.frame_center_x - self.line_center_deadzone_pixels, self.frame_height), (0, 0, 255), 1)
            cv2.line(display_frame, (self.frame_center_x + self.line_center_deadzone_pixels, 0),
                       (self.frame_center_x + self.line_center_deadzone_pixels, self.frame_height), (0, 0, 255), 1)

            # 顯示狀態文字
            state_text_map = {
                STATE_MARATHON_IDLE: "IDLE",
                STATE_MARATHON_FOLLOWING_LINE: "FOLLOWING_LINE",
                STATE_MARATHON_LINE_LOST: "LINE_LOST",
                STATE_MARATHON_COMPLETED: "COMPLETED/STOPPED"
            }
            cv2.putText(display_frame, f"State: {state_text_map.get(self.current_state, 'Unknown')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {(current_time - self.marathon_start_time):.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Marathon Competition (Color Only)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("使用者提前終止競賽。")
                self.current_state = STATE_MARATHON_COMPLETED # 標記為結束
                self.ser.send_command(MOVE_COMMAND, [0])
                break
            elif key == ord('r'): # 手動重置回跟隨線條狀態 (測試用)
                print("手動重置為 FOLLOW_LINE 狀態。")
                self.current_state = STATE_MARATHON_FOLLOWING_LINE
                self.last_line_seen_time = time.time()


        print("馬拉松結束，釋放資源。")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if self.ser and self.ser.is_open:
            self.ser.send_command(MOVE_COMMAND, [0]) # 確保機器人停止
            self.ser.close()


def load_marathon_line_profile_only(filepath):
    profiles_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        loaded_profiles = data.get("profiles", [])
        found_track_line = False
        for pd in loaded_profiles:
            if pd.get("name") == PROFILE_NAME_TRACK_LINE: # 只尋找賽道線的設定檔
                profile_instance = ColorProfile.from_dict(pd)
                profiles_dict[profile_instance.name] = profile_instance
                print(f"  已載入賽道線設定檔: '{profile_instance.name}' (啟用: {profile_instance.active})")
                found_track_line = True
                break # 只需要這一個
        if not found_track_line:
             print(f"警告：未在 '{filepath}' 中找到賽道線設定檔 '{PROFILE_NAME_TRACK_LINE}'。")
             return None
    except FileNotFoundError:
        print(f"錯誤：找不到顏色設定檔 '{filepath}'。"); return None
    except Exception as e:
        print(f"載入顏色設定檔時發生錯誤：{e}"); return None
    return profiles_dict

if __name__ == "__main__":
    print("正在初始化 HuroCup 馬拉松競賽程式 (僅顏色偵測直線)...")

    line_color_profile = load_marathon_line_profile_only(PROFILE_FILE_NAME_MARATHON)
    if not line_color_profile:
        print("無法載入賽道線顏色設定檔，程式結束。")
        exit()

    serial_controller = SerialCommunication(port=SERIAL_PORT, baudrate=BAUDRATE)
    if not serial_controller.ser or not serial_controller.ser.is_open:
        available_ports = SerialCommunication.get_available_ports()
        print(f"無法連接序列埠。可用埠口：{available_ports if available_ports else '無'}")
        exit()

    marathon_runner = MarathonRobotColorOnly(serial_controller, line_color_profile)
    if not marathon_runner.initialize_camera(CAMERA_INDEX):
        serial_controller.close(); exit()

    try:
        marathon_runner.run_marathon_logic()
    except KeyboardInterrupt:
        print("\n使用者中斷程式執行。")
    finally:
        print("正在關閉程式...")
        if marathon_runner.cap and marathon_runner.cap.isOpened():
            marathon_runner.cap.release()
        cv2.destroyAllWindows()
        if serial_controller.ser and serial_controller.ser.is_open:
            serial_controller.send_command(MOVE_COMMAND, [0])
            serial_controller.close()
        print("程式已關閉。")