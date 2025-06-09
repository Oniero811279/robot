import cv2
import numpy as np
import time
import json
import serial
import serial.tools.list_ports
import platform # 用於偵測作業系統

# 嘗試從 HSV.py 匯入必要的類別。
# 假設 HSV.py 中定義了 SerialCommunication 和 OpenCVThread 類別。
# 如果 HSV.py 還有其他您希望共享的常數或函數，也可以在這裡匯入。
try:
    from HSV import SerialCommunication, OpenCVThread
    # 由於原 HSV.py 主要是 GUI，可能沒有明確定義競賽狀態常數，
    # 這些將在這裡定義。
except ImportError as e:
    print(f"錯誤：無法從 HSV.py 匯入必要的類別。請確認 HSV.py 的名稱和路徑正確，且其內部定義了 SerialCommunication 和 OpenCVThread 類別。錯誤訊息: {e}")
    print("請確保 HSV.py 在本程式碼的同一個目錄下，或者在 Python 的模組搜索路徑中。")
    exit()

# --- 顏色設定檔類別定義 ---
# 這個類別用於解析和儲存從 JSON 檔案載入的顏色範圍資訊
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

# --- 競賽參數與設定 ---
CAMERA_INDEX = 0  # 請依實際情況調整攝影機索引
COLOR_RANGE_FILE = "sprint.json" # 儲存目標顏色設定檔的檔案名稱
SERIAL_PORT ="/dev/ttyACM0" # 例如 "COM3" (Windows) 或 "/dev/ttyUSB0" (Linux/macOS)
                    # 設為 None 時，程式會嘗試列出可用串口供選擇，但競賽模式通常需要預設。
BAUDRATE = 1000000 # 機器人通訊鮑率

# 目標尺寸閾值 (像素面積)
# 當目標面積達到這個值時，機器人會認為足夠接近
TARGET_AREA_THRESHOLD_STOP = 20000 # 假設目標大於20000像素面積時停止前進

# 機器人步長和角度參數
INITIAL_FORWARD_STEP = 50       # 初始前進步長
MIN_FORWARD_STEP = 5            # 最小前進步長 (避免完全停止前進)
REVERSE_STEP = -30              # 後退步長 (負號表示後退)
MAX_TURN_ANGLE = 10             # 最大轉彎角度 (左右轉的最大角度)
MAX_SIDE_STEP = 20              # 最大平行走步長 (左右平移的最大步長)

# 停止後退的條件
REVERSE_DURATION = 10 # 後退3秒鐘 (可根據實際比賽規則調整)

# --- 競賽狀態定義 ---
# 這些狀態將幫助機器人管理其行為流程
SPRINT_STATE_IDLE = 0            # 閒置/等待開始
SPRINT_STATE_FORWARD_RUN = 1     # 向前追蹤目標
SPRINT_STATE_REACHED_TARGET = 2  # 已達到目標 (準備後退)
SPRINT_STATE_BACKWARD_RUN = 3    # 向後行駛
SPRINT_STATE_COMPLETED = 4       # 競賽完成
SPRINT_STATE_FOUL = 5            # 犯規/異常狀態

# --- Sprint 機器人主控邏輯 ---
class SprintRobot:
    def __init__(self, serial_comm: SerialCommunication, camera_index=0):
        self.serial_comm = serial_comm
        self.camera_index = camera_index
        self.opencv_thread = None # 將在 initialize_camera 中實例化
        self.target_color_range = None # 將在載入文件後設置

        self.current_state = SPRINT_STATE_IDLE
        self.reverse_start_time = 0

        self.current_target_center = None
        self.current_target_area = 0
        self.current_frame_width = 0 # 來自 OpenCVThread 的實際幀寬度

        self.competition_running = False # 控制主迴圈是否執行競賽邏輯

    def initialize_camera(self):
        """
        初始化 OpenCVThread 並連接信號。
        """
        print(f"初始化攝影機 (索引: {self.camera_index})...")
        self.opencv_thread = OpenCVThread(self.camera_index)

        # 連接來自 OpenCVThread 的信號
        # 我們不需要 signal_frame 在這裡顯示，但我們可以接收它來更新內部狀態。
        # 如果您想在 SprintCompetition.py 中也顯示圖像，您可以在這裡處理 signal_frame。
        # 例如：self.opencv_thread.signal_frame.connect(self.display_image)
        self.opencv_thread.signal_target_data.connect(self._update_vision_data)

        # 嘗試載入顏色範圍並設定到 OpenCVThread
        if not self._load_and_set_color_ranges(COLOR_RANGE_FILE):
            print("警告: 未能成功載入或設置顏色範圍。機器人可能無法偵測目標。")
            return False

        self.opencv_thread.start() # 啟動 OpenCV 線程
        print("攝影機初始化成功。")
        return True

    def _load_and_set_color_ranges(self, filename):
        """
        從 JSON 文件載入 ColorProfile 並應用到 OpenCVThread。
        該檔案可能包含單個 ColorProfile 物件，或者是一個包含多個 ColorProfile 物件的列表。
        如果為列表，則優先選擇 'active' 為 True 的第一個設定檔；如果都沒有 active，則選擇列表中的第一個。
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
                selected_profile = None

                if isinstance(data, list):
                    # 如果是列表，嘗試找到 'active' 為 True 的設定檔
                    for profile_data in data:
                        profile = ColorProfile.from_dict(profile_data)
                        if profile.active:
                            selected_profile = profile
                            break # 找到第一個 active 的就停止
                    
                    # 如果列表中沒有 active 的設定檔，則使用列表中的第一個
                    if selected_profile is None and data:
                        selected_profile = ColorProfile.from_dict(data[0])
                    elif selected_profile is None and not data:
                        print(f"錯誤: 顏色範圍文件 '{filename}' 為空列表。")
                        return False
                elif isinstance(data, dict):
                    # 如果是單個字典，直接解析為 ColorProfile
                    selected_profile = ColorProfile.from_dict(data)
                else:
                    print(f"錯誤: '{filename}' 文件的格式不正確。期望是 ColorProfile 物件 (字典) 或包含 ColorProfile 物件的列表。")
                    return False

                if selected_profile:
                    # 從選定的 ColorProfile 中提取 HSV 範圍
                    lower = [selected_profile.c1_min, selected_profile.c2_min, selected_profile.c3_min]
                    upper = [selected_profile.c1_max, selected_profile.c2_max, selected_profile.c3_max]
                    self.target_color_range = {"lower": lower, "upper": upper}
                    self.opencv_thread.set_hsv_range(lower, upper)
                    print(f"從 {filename} 載入並設置 HSV 範圍成功。目標顏色設定檔: '{selected_profile.name}'")
                    return True
                else:
                    print(f"錯誤: 在 '{filename}' 中找不到可用的顏色設定檔。")
                    return False

        except FileNotFoundError:
            print(f"錯誤: 找不到顏色範圍文件 '{filename}'。請確認檔案是否存在且路徑正確。")
            return False
        except json.JSONDecodeError:
            print(f"錯誤: 無法解析 '{filename}'。請檢查檔案內容是否為有效的 JSON 格式。")
            return False
        except Exception as e:
            print(f"載入顏色範圍文件時發生錯誤: {e}")
            return False

    def _update_vision_data(self, target_data):
        """
        接收來自 OpenCVThread 的目標數據，並更新機器人的視覺狀態。
        target_data: (target_center, target_area, bbox, frame_width, frame_height)
        """
        self.current_target_center, self.current_target_area, _, self.current_frame_width, _ = target_data
        # print(f"視覺數據更新: 中心={self.current_target_center}, 面積={self.current_target_area}") # 可選：用於調試

    def _calculate_robot_commands(self):
        """
        根據當前競賽狀態和視覺數據計算機器人指令。
        回傳: forward_step, turn_angle, side_step
        """
        forward_step = 0
        turn_angle = 0
        side_step = 0

        # 如果沒有有效的幀寬度，則停止所有動作
        if self.current_frame_width == 0:
            return 0, 0, 0

        frame_center_x = self.current_frame_width // 2

        if self.current_state == SPRINT_STATE_FORWARD_RUN:
            if self.current_target_center and self.current_target_area > 0:
                if self.current_target_area > TARGET_AREA_THRESHOLD_STOP:
                    # 目標達到閾值，切換到後退狀態
                    print(f"目標面積達到閾值 ({self.current_target_area} > {TARGET_AREA_THRESHOLD_STOP})，切換到後退狀態。")
                    self.current_state = SPRINT_STATE_REACHED_TARGET
                    self.reverse_start_time = time.time() # 記錄開始後退的時間
                    forward_step = 0 # 立即停止前進
                    turn_angle = 0
                    side_step = 0
                else:
                    # 向前追蹤目標，根據面積調整速度
                    # normalized_area = min(self.current_target_area / (TARGET_AREA_THRESHOLD_STOP * 1.5), 1.0)
                    # # forward_step = INITIAL_FORWARD_STEP - (INITIAL_FORWARD_STEP - MIN_FORWARD_STEP) * normalized_area
                    # # forward_step = int(max(MIN_FORWARD_STEP, forward_step))
                    forward_step = 35

                    # 根據中心點偏移調整轉彎角度和側移步長
                    offset_x = self.current_target_center[0] - frame_center_x
                    normalized_offset_x = offset_x / (self.current_frame_width / 2) # 歸一化偏移量

                    turn_angle = -int(normalized_offset_x * MAX_TURN_ANGLE)
                    # side_step = -int(normalized_offset_x * MAX_SIDE_STEP)

                    turn_angle = max(-MAX_TURN_ANGLE, min(MAX_TURN_ANGLE, turn_angle))
                    # side_step = max(-MAX_SIDE_STEP, min(MAX_SIDE_STEP, side_step))
                    
                    turn_angle = max(-MAX_TURN_ANGLE, min(MAX_TURN_ANGLE, turn_angle))
                    side_step = 0
            else:
                # 前進狀態但未偵測到目標，停止或原地搜索
                print("前進狀態但未偵測到目標，機器人停止。")
                forward_step = 0
                turn_angle = 0
                side_step = 0

        elif self.current_state == SPRINT_STATE_REACHED_TARGET:
            # 停止等待，準備切換到後退
            forward_step = 0
            turn_angle = 0
            side_step = 0
            # 這裡可以加入一個短暫的停頓，確保完全停止後再後退
            if time.time() - self.reverse_start_time > 0.5: # 停頓 0.5 秒
                self.current_state = SPRINT_STATE_BACKWARD_RUN
                self.reverse_start_time = time.time() # 重置後退計時

        elif self.current_state == SPRINT_STATE_BACKWARD_RUN:
            forward_step = REVERSE_STEP
            if time.time() - self.reverse_start_time >= REVERSE_DURATION:
                # 後退時間結束，競賽完成
                print("後退完成，競賽結束。")
                self.current_state = SPRINT_STATE_COMPLETED
                forward_step = 0
                turn_angle = 0
                side_step = 0
            # 如果後退時也需要橫向調整，可以在這裡加入視覺判斷
            # 例如：檢查機器人是否偏離了後退的直線路徑 (如果比賽有起點線或指引線)

        elif self.current_state == SPRINT_STATE_IDLE:
            # 閒置狀態，等待開始指令
            pass

        elif self.current_state == SPRINT_STATE_COMPLETED:
            # 競賽完成，機器人停止
            pass

        elif self.current_state == SPRINT_STATE_FOUL:
            # 犯規狀態，機器人停止
            pass

        return forward_step, turn_angle, side_step

    def start_competition(self):
        """
        啟動機器人競賽主流程。
        """
        if not self.serial_comm.ser or not self.serial_comm.ser.is_open:
            print("錯誤：串口未連接或已關閉。請先連接串口。")
            return False
        if not self.opencv_thread or not self.opencv_thread.running:
            print("錯誤：攝影機線程未運行或初始化失敗。")
            return False
        if not self.target_color_range:
            print("錯誤：未載入目標顏色範圍。請先確認 color_ranges.json 存在並正確設置。")
            return False

        self.current_state = SPRINT_STATE_FORWARD_RUN
        self.competition_running = True
        print("機器人競賽開始！進入向前追蹤狀態。")
        return True

    def stop_competition(self):
        """
        停止機器人競賽主流程並發送停止指令。
        """
        self.competition_running = False
        self.current_state = SPRINT_STATE_IDLE
        self.serial_comm.send_command("run,0,0,0\n") # 發送停止指令
        print("機器人競賽停止。")

    def run(self):
        """
        機器人競賽的主迴圈。
        """
        print("等待競賽開始指令...")
        # 這裡可以加入一個循環，等待外部觸發 (例如按鍵、接收特定串口指令) 啟動競賽
        # 為了簡化，我們直接在這裡啟動競賽，或者您可以手動調用 start_competition()

        # 等待攝像頭和串口初始化完成
        time.sleep(3) # 給予足夠時間

        if not self.start_competition():
            print("競賽啟動失敗。")
            self._cleanup()
            return

        # 主控制迴圈
        while self.competition_running or self.current_state == SPRINT_STATE_BACKWARD_RUN:
            forward_step, turn_angle, side_step = self._calculate_robot_commands()
            command_str = f"run,{forward_step},{turn_angle},{side_step}\n"
            self.serial_comm.send_command(command_str)
            # print(f"狀態: {self.current_state}, 指令: {command_str.strip()}") # 用於調試

            if self.current_state == SPRINT_STATE_COMPLETED:
                print("競賽已完成，停止運行。")
                self.stop_competition()
                break # 退出主循環

            time.sleep(0.05) # 控制迴圈頻率，例如 20 FPS (每 50ms 執行一次)

        print("競賽程式結束。")
        self._cleanup()

    def _cleanup(self):
        """清理資源"""
        if self.opencv_thread:
            self.opencv_thread.stop()
        if self.serial_comm:
            self.serial_comm.close()
        cv2.destroyAllWindows() # 關閉所有 OpenCV 窗口 (如果有的話)


# --- 主執行區塊 ---
if __name__ == "__main__":
    # 1. 偵測並選擇串口 (僅在沒有明確指定 SERIAL_PORT 時)
    if SERIAL_PORT is None:
        print("正在偵測可用串口...")
        ports = SerialCommunication.list_available_ports()
        if not ports:
            print("錯誤：未偵測到任何可用串口。程式即將結束。")
            exit()
        else:
            print("可用串口：")
            for i, (port, desc, hwid) in enumerate(ports):
                print(f"[{i}] {port}: {desc}")
            selected_index = -1
            while selected_index < 0 or selected_index >= len(ports):
                try:
                    user_input = input(f"請輸入要連接的串口編號 (0-{len(ports)-1}): ")
                    selected_index = int(user_input)
                except ValueError:
                    print("輸入無效，請輸入數字。")
            SERIAL_PORT = ports[selected_index][0]
            print(f"已選擇串口：{SERIAL_PORT}")
    else:
        print(f"使用預設串口：{SERIAL_PORT}")

    # 2. 初始化序列通訊
    print(f"正在初始化序列通訊 (埠：{SERIAL_PORT}, 鮑率：{BAUDRATE})...")
    serial_controller = SerialCommunication(port=SERIAL_PORT, baudrate=BAUDRATE)
    if not serial_controller.ser or not serial_controller.ser.is_open:
        print(f"錯誤：無法連接到序列埠 {SERIAL_PORT}。請檢查機器人連線並確認埠設定。程式即將結束。")
        exit()
    print("序列通訊初始化成功。")

    # 3. 建立並執行 SprintRobot
    sprint_robot = SprintRobot(serial_controller, CAMERA_INDEX)
    if not sprint_robot.initialize_camera():
        print("攝影機初始化失敗，程式即將結束。")
        serial_controller.close()
        exit()

    # 為了方便在競賽開始前準備，程式在這裡等待使用者輸入
    input("按下 Enter 鍵啟動機器人競賽...")
    sprint_robot.run() # 啟動競賽主迴圈