import cv2
import numpy as np
import time
import json
import serial
import serial.tools.list_ports
import platform
import threading
from pynput import keyboard  # <--- ADDED: Import keyboard listener

# --- Import necessary classes from the GUI application file ---
try:
    from HSV11 import SerialCommunication, ColorProfile
except ImportError:
    print("Error: Could not import from HSV11.py.")
    print("Please ensure 'HSV11.py' is in the same directory as this script.")
    exit()

# --- Competition Parameters & Settings ---
CAMERA_INDEX = 0
COLOR_RANGE_FILE = "sprint.json"
SERIAL_PORT = None 
BAUDRATE = 1000000

# --- Thresholds and Movement Parameters ---
TARGET_AREA_THRESHOLD_STOP = 25000 
REVERSE_DURATION_S = 2

# --- Robot Movement Parameters ---
FORWARD_SPEED = 35
MAX_TURN_SPEED = 15
REVERSE_SPEED = -20

# --- Competition State Definitions ---
STATE_IDLE = 0
STATE_SEARCHING = 1
STATE_FORWARD_RUN = 2
STATE_REACHED_TARGET = 3
STATE_BACKWARD_RUN = 4
STATE_COMPLETED = 5

# --- New Vision Thread for Standalone Operation ---
class VisionThread(threading.Thread):
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.lock = threading.Lock()
        self._target_data = {
            "center": None,
            "area": 0,
            "frame_width": 0,
            "frame_height": 0
        }
        self._color_settings = {
            "lower": np.array([0, 0, 0]),
            "upper": np.array([179, 255, 255]),
            "space": 'HSV'
        }
        self.daemon = True

    def set_color_properties(self, lower, upper, color_space='HSV'):
        with self.lock:
            self._color_settings["lower"] = np.array(lower)
            self._color_settings["upper"] = np.array(upper)
            self._color_settings["space"] = color_space
        print(f"VisionThread: Color properties set for space {color_space}")

    def get_target_data(self):
        with self.lock:
            return self._target_data.copy()

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera at index {self.camera_index}")
            self.running = False
            return

        print("VisionThread: Camera opened successfully. Starting loop.")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            
            with self.lock:
                lower_bound = self._color_settings["lower"]
                upper_bound = self._color_settings["upper"]
                color_space = self._color_settings["space"]

            if color_space == 'HSV':
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif color_space == 'LAB':
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            elif color_space == 'YCrCb':
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            elif color_space == 'HLS':
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            else:
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(converted_frame, lower_bound, upper_bound)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_center = None
            target_area = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        target_center = (cx, cy)
                        target_area = area
            
            with self.lock:
                self._target_data["center"] = target_center
                self._target_data["area"] = target_area
                self._target_data["frame_width"] = w
                self._target_data["frame_height"] = h

            time.sleep(0.01)

        cap.release()
        cv2.destroyAllWindows()
        print("VisionThread: Stopped and resources released.")


# --- Main Sprint Robot Controller Logic ---
class SprintRobot:
    def __init__(self, serial_comm, camera_index=0):
        self.serial_comm = serial_comm
        self.vision_thread = VisionThread(camera_index)
        self.current_state = STATE_IDLE
        self.state_timer = 0
        self.competition_running = False

    def initialize(self):
        print("Initializing Robot...")
        if not self._load_and_set_color_ranges(COLOR_RANGE_FILE):
            print("CRITICAL: Failed to load color ranges. Robot cannot operate.")
            return False
        
        self.vision_thread.start()
        time.sleep(2)
        print("Robot Initialized Successfully.")
        return True

    def _load_and_set_color_ranges(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            profiles_data = data.get("profiles", [])
            if not profiles_data:
                print(f"Error: No profiles found in '{filename}'.")
                return False

            selected_profile_data = None
            for p in profiles_data:
                if p.get("active", False):
                    selected_profile_data = p
                    break
            if not selected_profile_data:
                selected_profile_data = profiles_data[0]

            profile = ColorProfile.from_dict(selected_profile_data)
            color_space = data.get("current_color_space", "HSV")
            
            lower = [profile.c1_min, profile.c2_min, profile.c3_min]
            upper = [profile.c1_max, profile.c2_max, profile.c3_max]

            self.vision_thread.set_color_properties(lower, upper, color_space)
            print(f"Successfully loaded profile '{profile.name}' from '{filename}'.")
            return True

        except FileNotFoundError:
            print(f"Error: Color settings file not found: '{filename}'")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{filename}'.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while loading settings: {e}")
            return False

    def start_competition(self):
        if not self.serial_comm.ser or not self.serial_comm.ser.is_open:
            print("Error: Serial port not connected.")
            return
        if not self.vision_thread.is_alive():
            print("Error: Vision thread is not running.")
            return
        
        print("\n--- Competition Starting! ---")
        self.competition_running = True
        self.current_state = STATE_SEARCHING

    def stop_competition(self):
        # This check prevents stopping actions from running multiple times
        if self.competition_running:
            print("--- Stop signal received. Halting competition. ---")
            self.competition_running = False

    def run_main_loop(self):
        while self.competition_running:
            vision_data = self.vision_thread.get_target_data()
            self._execute_state_machine(vision_data)
            time.sleep(0.05)
        
        # Ensure robot is stopped when loop exits
        print("Main loop has exited. Sending final stop command.")
        self.serial_comm.send_command("run", [0, 0, 0])


    def _execute_state_machine(self, vision_data):
        target_center = vision_data["center"]
        target_area = vision_data["area"]
        frame_width = vision_data["frame_width"]

        fwd = 0
        turn = 0
        
        if self.current_state == STATE_SEARCHING:
            print("State: SEARCHING... looking for target.")
            if target_center:
                print("Target found. Switching to FORWARD_RUN.")
                self.current_state = STATE_FORWARD_RUN
            else:
                fwd = 0
                turn = 5 

        elif self.current_state == STATE_FORWARD_RUN:
            if not target_center or frame_width == 0:
                print("Target lost. Switching back to SEARCHING.")
                self.current_state = STATE_SEARCHING
                return

            if target_area > TARGET_AREA_THRESHOLD_STOP:
                print(f"Target Reached! Area: {target_area}. Switching to REACHED_TARGET.")
                self.current_state = STATE_REACHED_TARGET
                return
            
            frame_center_x = frame_width / 2
            offset = target_center[0] - frame_center_x
            normalized_offset = offset / frame_center_x 
            
            fwd = FORWARD_SPEED
            turn = -int(normalized_offset * MAX_TURN_SPEED)
            
            print(f"State: FORWARD_RUN | Area: {target_area:5.0f} | Turn: {turn:3d}")

        elif self.current_state == STATE_REACHED_TARGET:
            print("State: REACHED_TARGET. Preparing to reverse.")
            self.state_timer = time.time()
            self.current_state = STATE_BACKWARD_RUN
            fwd = 0
            turn = 0

        elif self.current_state == STATE_BACKWARD_RUN:
            if time.time() - self.state_timer > REVERSE_DURATION_S:
                print("Reverse complete. Switching to COMPLETED.")
                self.current_state = STATE_COMPLETED
                return
            
            print(f"State: BACKWARD_RUN")
            fwd = REVERSE_SPEED
            turn = 0
        
        elif self.current_state == STATE_COMPLETED:
            print("State: COMPLETED. Mission finished.")
            fwd = 0
            turn = 0
            self.stop_competition()

        self.serial_comm.send_command("run", [fwd, turn, 0])

    def cleanup(self):
        print("Cleaning up resources...")
        if self.vision_thread.is_alive():
            self.vision_thread.stop()
            self.vision_thread.join(timeout=2)
        if self.serial_comm:
            self.serial_comm.close()
        print("Cleanup complete.")

# --- Main Execution Block ---
if __name__ == "__main__":
    if SERIAL_PORT is None:
        print("Detecting available serial ports...")
        available_ports = SerialCommunication.get_available_ports()
        if not available_ports:
            print("ERROR: No serial ports detected. Please connect the robot.")
            exit()
        
        print("Available ports:")
        for i, port in enumerate(available_ports):
            print(f"  [{i}] {port}")
        
        selected_index = -1
        while not (0 <= selected_index < len(available_ports)):
            try:
                choice = input(f"Select port number (0-{len(available_ports)-1}): ")
                selected_index = int(choice)
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid number.")
        SERIAL_PORT = available_ports[selected_index]

    print(f"Connecting to {SERIAL_PORT} at {BAUDRATE}")