#!/usr/bin/env python3
"""
Security Monitor - Face-based Screen Lock
Monitors webcam for authorized faces and locks screen when unauthorized access is detected.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
import os
import sys
import pickle
import json
from collections import deque
import threading
# Optional Windows UI helpers (safe to run without them)
try:
    import win32api
    import win32con
    import win32gui
    WIN32_AVAILABLE = True
except Exception:
    WIN32_AVAILABLE = False
"""
Auto-password features disabled: focusing on lock-out only.
"""

class SecurityMonitor:
    def __init__(self):
        self.face_database = {}
        self.liveness_state = {
            "ear_history": deque(maxlen=30),
            "mouth_history": deque(maxlen=30),
            "head_pose_history": deque(maxlen=30),
            "blink_count": 0,
            "last_blink_frame": 0
        }
        
        # Security settings
        self.lock_delay = 10  # seconds before locking computer
        self.unlock_delay = 1  # seconds before unlocking
        self.similarity_threshold = 0.90  # 90% accuracy as requested
        self.startup_delay = 10  # seconds to wait before monitoring starts
        self.user_present = False
        self.last_seen = 0
        self.screen_locked = False
        self.monitoring = False
        
        # Auto-login disabled
        self.auto_login_enabled = False
        self.password = None
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, 
            refine_landmarks=True, min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Load face database
        self.load_face_database()
        
        # Auto-login disabled; skip any password loading
        
    def load_face_database(self):
        """Load registered faces from database"""
        db_path = "face_database.pkl"
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"SUCCESS: Loaded {len(self.face_database)} registered faces")
            except Exception as e:
                print(f"ERROR: Error loading face database: {e}")
                self.face_database = {}
        else:
            print("WARNING: No face database found. Please register your face first.")
            self.face_database = {}
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        if len(eye_landmarks) < 6:
            return 0
        
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio for mouth movement detection"""
        if len(mouth_landmarks) < 20:
            return 0
        
        vertical_1 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])
        vertical_2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
        vertical_3 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
        
        mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
        return mar
    
    def extract_face_embedding(self, face_landmarks):
        """Extract face embedding from MediaPipe landmarks"""
        if not face_landmarks:
            return None
        
        embedding = []
        for landmark in face_landmarks.landmark:
            embedding.extend([landmark.x, landmark.y, landmark.z])
        
        # Ensure consistent embedding size (468 landmarks * 3 coordinates = 1404)
        embedding_array = np.array(embedding)
        if len(embedding_array) != 1404:
            if len(embedding_array) > 1404:
                embedding_array = embedding_array[:1404]  # Truncate if too long
            else:
                # Pad with zeros if too short
                padding = np.zeros(1404 - len(embedding_array))
                embedding_array = np.concatenate([embedding_array, padding])
        
        return embedding_array
    
    def calculate_face_similarity(self, embedding1, embedding2):
        """Calculate similarity between two face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def check_liveness(self, frame_count):
        """Check if the detected face is live (not a photo/video)"""
        liveness_checks = {
            "blink_detected": False,
            "mouth_movement": False,
            "micro_movements": False
        }
        
        # Check for blinks
        if len(self.liveness_state["ear_history"]) > 0:
            current_ear = self.liveness_state["ear_history"][-1]
            ear_threshold = 0.25
            consecutive_low = 0
            
            for recent_ear in list(self.liveness_state["ear_history"])[-3:]:
                if recent_ear < ear_threshold:
                    consecutive_low += 1
            
            if consecutive_low >= 2 and frame_count - self.liveness_state["last_blink_frame"] > 10:
                self.liveness_state["blink_count"] += 1
                self.liveness_state["last_blink_frame"] = frame_count
                liveness_checks["blink_detected"] = True
        
        # Check mouth movement
        if len(self.liveness_state["mouth_history"]) > 5:
            recent_mars = list(self.liveness_state["mouth_history"])[-5:]
            mar_variance = np.var(recent_mars)
            liveness_checks["mouth_movement"] = mar_variance > 0.001
        
        # Check for micro-movements
        if len(self.liveness_state["ear_history"]) > 10:
            ear_variance = np.var(list(self.liveness_state["ear_history"])[-10:])
            liveness_checks["micro_movements"] = ear_variance > 0.0001
        
        return liveness_checks
    
    def lock_screen(self):
        """Lock computer when unauthorized access detected"""
        if not self.screen_locked:
            print("LOCKING COMPUTER - Unauthorized access detected")
            try:
                # Lock the computer (log out user)
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], capture_output=True)
                self.screen_locked = True
                print("Computer locked (user logged out)")
            except Exception as e:
                print(f"ERROR: Error locking computer: {e}")
    
    
    def verify_face(self, face_embedding):
        """Verify if the detected face matches registered faces"""
        if face_embedding is None or not self.face_database:
            return False, 0.0
        
        best_similarity = 0.0
        best_match = None
        for name, face_data in self.face_database.items():
            similarity = self.calculate_face_similarity(face_embedding, face_data["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        verified = best_similarity > self.similarity_threshold
        if verified:
            print(f"SUCCESS: Face verified as {best_match} (similarity: {best_similarity:.2f})")
        else:
            print(f"FAILED: Face not verified (best similarity: {best_similarity:.2f}, threshold: {self.similarity_threshold:.2f})")
        
        return verified, best_similarity
    
    def monitor_webcam(self):
        """Main monitoring loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        print("SUCCESS: Webcam opened - camera light should be on now!")
        print("Starting security monitoring...")
        print("Press 'q' to quit, 'r' to reload face database")
        
        frame_count = 0
        last_fps_time = time.time()
        startup_time = time.time()
        
        print(f"Starting up - monitoring will begin in {self.startup_delay} seconds...")

        # Prepare small status window (text only)
        window_name = 'Security Monitor'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0)

        # Snapshot win32 availability into a local flag to avoid scope issues
        win32_avail = WIN32_AVAILABLE
        # Allow monitor selection and margins via environment variables
        try:
            env_monitor_index = int(os.environ.get("SECURITY_MONITOR_MONITOR_INDEX", "1"))
        except Exception:
            env_monitor_index = 1
        try:
            margin_x = int(os.environ.get("SECURITY_MONITOR_MARGIN_X", "30"))
        except Exception:
            margin_x = 30
        try:
            margin_y = int(os.environ.get("SECURITY_MONITOR_MARGIN_Y", "120"))
        except Exception:
            margin_y = 120

        # Get monitor geometry (second monitor if present)
        if win32_avail:
            try:
                mons = win32api.EnumDisplayMonitors()
                # mons: list of (hMonitor, hdcMonitor, rect)
                rects = [m[2] for m in mons] if mons else []
                if len(rects) > env_monitor_index:
                    mon = rects[env_monitor_index]
                elif len(rects) >= 1:
                    mon = rects[0]
                else:
                    mon = (0, 0, 800, 600)
                mon_w = mon[2] - mon[0]
                mon_h = mon[3] - mon[1]
            except Exception:
                win32_avail = False
                mon = (0, 0, 800, 600)
                mon_w, mon_h = 800, 600
        else:
            mon = (0, 0, 800, 600)
            mon_w, mon_h = 800, 600
        # Text-only window aligned bottom-right; compact, terminal-like text
        text_w = 280
        line_h = 14
        window_w = text_w
        window_h = line_h + 4
        cv2.resizeWindow(window_name, window_w, window_h)

        # Position bottom-right of selected monitor
        # Align window bottom-right of monitor
        pos_x = mon[0] + mon_w - window_w - max(0, margin_x)
        # Raise the window by margin_y from bottom to ensure visibility above taskbar/docks
        pos_y = mon[1] + mon_h - window_h - max(0, margin_y)
        # Clamp into monitor bounds
        pos_x = max(mon[0] + 2, min(pos_x, mon[0] + mon_w - window_w - 2))
        pos_y = max(mon[1] + 2, min(pos_y, mon[1] + mon_h - window_h - 2))
        try:
            cv2.moveWindow(window_name, pos_x, pos_y)
        except Exception:
            pass

        # If Win32 available, remove window chrome and set colorkey transparency (black)
        if win32_avail:
            try:
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd:
                    GWL_STYLE = -16
                    GWL_EXSTYLE = -20
                    WS_CAPTION = 0x00C00000
                    WS_THICKFRAME = 0x00040000
                    WS_MINIMIZEBOX = 0x00020000
                    WS_MAXIMIZEBOX = 0x00010000
                    WS_SYSMENU = 0x00080000
                    style = win32gui.GetWindowLong(hwnd, GWL_STYLE)
                    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
                    win32gui.SetWindowLong(hwnd, GWL_STYLE, style)

                    WS_EX_LAYERED = 0x00080000
                    ex = win32gui.GetWindowLong(hwnd, GWL_EXSTYLE)
                    ex |= WS_EX_LAYERED
                    win32gui.SetWindowLong(hwnd, GWL_EXSTYLE, ex)

                    LWA_COLORKEY = 0x00000001
                    # Set black as transparent colorkey
                    win32gui.SetLayeredWindowAttributes(hwnd, 0x000000, 0, LWA_COLORKEY)
            except Exception:
                pass
        
        while self.monitoring:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read from webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Check if we're still in startup delay period
            if (current_time - startup_time) < self.startup_delay:
                remaining_startup = self.startup_delay - (current_time - startup_time)
                print(f"Startup delay: {remaining_startup:.1f}s remaining...")
                time.sleep(0.5)
                continue
            
            # Run at 2 FPS (500ms delay between frames)
            time.sleep(0.5)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and landmarks
            face_detection_results = self.face_detection.process(rgb_frame)
            face_mesh_results = self.face_mesh.process(rgb_frame)
            
            face_detected = False
            face_verified = False
            face_similarity = 0.0
            
            if face_detection_results.detections and face_mesh_results.multi_face_landmarks:
                for i, detection in enumerate(face_detection_results.detections):
                    if i < len(face_mesh_results.multi_face_landmarks):
                        landmarks = face_mesh_results.multi_face_landmarks[i]
                        
                        # Extract facial features for liveness detection
                        h, w, _ = frame.shape
                        left_eye_points = np.array([
                            [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
                            [landmarks.landmark[7].x * w, landmarks.landmark[7].y * h],
                            [landmarks.landmark[163].x * w, landmarks.landmark[163].y * h],
                            [landmarks.landmark[144].x * w, landmarks.landmark[144].y * h],
                            [landmarks.landmark[145].x * w, landmarks.landmark[145].y * h],
                            [landmarks.landmark[153].x * w, landmarks.landmark[153].y * h]
                        ])
                        
                        right_eye_points = np.array([
                            [landmarks.landmark[362].x * w, landmarks.landmark[362].y * h],
                            [landmarks.landmark[382].x * w, landmarks.landmark[382].y * h],
                            [landmarks.landmark[381].x * w, landmarks.landmark[381].y * h],
                            [landmarks.landmark[380].x * w, landmarks.landmark[380].y * h],
                            [landmarks.landmark[374].x * w, landmarks.landmark[374].y * h],
                            [landmarks.landmark[373].x * w, landmarks.landmark[373].y * h]
                        ])
                        
                        mouth_points = np.array([
                            [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],
                            [landmarks.landmark[84].x * w, landmarks.landmark[84].y * h],
                            [landmarks.landmark[17].x * w, landmarks.landmark[17].y * h],
                            [landmarks.landmark[314].x * w, landmarks.landmark[314].y * h],
                            [landmarks.landmark[405].x * w, landmarks.landmark[405].y * h],
                            [landmarks.landmark[320].x * w, landmarks.landmark[320].y * h],
                            [landmarks.landmark[307].x * w, landmarks.landmark[307].y * h],
                            [landmarks.landmark[375].x * w, landmarks.landmark[375].y * h],
                            [landmarks.landmark[321].x * w, landmarks.landmark[321].y * h],
                            [landmarks.landmark[308].x * w, landmarks.landmark[308].y * h],
                            [landmarks.landmark[324].x * w, landmarks.landmark[324].y * h],
                            [landmarks.landmark[318].x * w, landmarks.landmark[318].y * h],
                            [landmarks.landmark[13].x * w, landmarks.landmark[13].y * h],
                            [landmarks.landmark[82].x * w, landmarks.landmark[82].y * h],
                            [landmarks.landmark[81].x * w, landmarks.landmark[81].y * h],
                            [landmarks.landmark[80].x * w, landmarks.landmark[80].y * h],
                            [landmarks.landmark[78].x * w, landmarks.landmark[78].y * h],
                            [landmarks.landmark[95].x * w, landmarks.landmark[95].y * h],
                            [landmarks.landmark[88].x * w, landmarks.landmark[88].y * h],
                            [landmarks.landmark[178].x * w, landmarks.landmark[178].y * h]
                        ])
                        
                        # Calculate liveness metrics
                        left_ear = self.calculate_ear(left_eye_points)
                        right_ear = self.calculate_ear(right_eye_points)
                        avg_ear = (left_ear + right_ear) / 2
                        mar = self.calculate_mouth_aspect_ratio(mouth_points)
                        
                        # Store in liveness state
                        self.liveness_state["ear_history"].append(avg_ear)
                        self.liveness_state["mouth_history"].append(mar)
                        
                        # Check liveness
                        liveness_checks = self.check_liveness(frame_count)
                        
                        # Extract face embedding and verify
                        face_embedding = self.extract_face_embedding(landmarks)
                        face_verified, face_similarity = self.verify_face(face_embedding)
                        
                        # Face is authorized if detected and verified
                        # Liveness checks are optional for now (too strict)
                        face_detected = True
                        user_authorized = face_verified
                        
                        break
            
            # Update security state
            if face_detected and user_authorized:
                # Authorized user detected
                if not self.user_present:
                    self.user_present = True
                    print(f"AUTHORIZED USER DETECTED (similarity: {face_similarity:.1%})")
                
                # Update last seen time
                self.last_seen = current_time
                
                # Only update presence; no auto-unlock
                pass
            else:
                # No face or unauthorized face detected
                if self.user_present:
                    self.user_present = False
                    if face_detected:
                        print(f"UNAUTHORIZED ACCESS DETECTED (similarity: {face_similarity:.1%})")
                    else:
                        print("NO FACE DETECTED")
                
                # Lock screen if user has been absent for lock_delay
                if not self.screen_locked and self.last_seen > 0 and (current_time - self.last_seen) > self.lock_delay:
                    self.lock_screen()
            
            # Calculate FPS
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                current_fps_time = time.time()
                fps = 10 / (current_fps_time - last_fps_time)
                last_fps_time = current_fps_time
            else:
                fps = 2.0  # Target FPS
            
            # Display status
            if (current_time - startup_time) < self.startup_delay:
                remaining_startup = self.startup_delay - (current_time - startup_time)
                status_text = f"STARTUP DELAY: {remaining_startup:.1f}s remaining..."
            else:
                status_text = f"Status: {'AUTHORIZED' if self.user_present else 'UNAUTHORIZED'}"
                if face_detected:
                    status_text += f" | Similarity: {face_similarity:.1%}"
                elif self.screen_locked:
                    status_text += " | LOCKED"
                elif not self.user_present and self.last_seen > 0:
                    time_remaining = self.lock_delay - (current_time - self.last_seen)
                    if time_remaining > 0:
                        status_text += f" | Lock in: {time_remaining:.1f}s"
                status_text += f" | FPS: {fps:.1f}"
            
            # Compose a transparent-friendly canvas (black=transparent) with text only
            canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
            txt_color = (0, 255, 0) if self.user_present else (0, 0, 255)
            y0 = window_h - 4
            # Only show "AUTHORIZED (xx.x%)" or "UNAUTHORIZED (xx.x%)"; or FAILURE if something broke
            try:
                if face_detected:
                    label = f"AUTHORIZED ({face_similarity:.1%})" if self.user_present else f"UNAUTHORIZED ({face_similarity:.1%})"
                else:
                    label = "UNAUTHORIZED (0.0%)"
            except Exception:
                label = "FAILURE"
            # Use a terminal-like font style: Hershey Plain, small scale, thin stroke
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1.0
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            x_right = window_w - 6
            x = max(4, x_right - tw)
            # Baseline adjustment for bottom placement
            cv2.putText(canvas, label, (x, y0), font, font_scale, txt_color, thickness, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)

            # Try keep window at bottom of Z-order (behind other windows)
            if win32_avail:
                try:
                    hwnd = win32gui.FindWindow(None, window_name)
                    if hwnd:
                        win32gui.SetWindowPos(
                            hwnd,
                            win32con.HWND_BOTTOM,
                            pos_x,
                            pos_y,
                            window_w,
                            window_h,
                            win32con.SWP_NOACTIVATE | win32con.SWP_NOSIZE | win32con.SWP_NOMOVE,
                        )
                except Exception:
                    pass
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.load_face_database()
                print("Face database reloaded")
            # Auto-login hotkey removed
        
        cap.release()
        cv2.destroyAllWindows()
    
    def start_monitoring(self):
        """Start the security monitoring"""
        if not self.face_database:
            print("ERROR: No registered faces found. Please register your face first using the web interface.")
            return
        
        self.monitoring = True
        try:
            self.monitor_webcam()
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.monitoring = False
            if self.screen_locked:
                self.unlock_screen()

def main():
    print("Security Monitor - Face-based Screen Lock")
    print("=" * 50)
    
    monitor = SecurityMonitor()
    
    if not monitor.face_database:
        print("\nTo register your face:")
        print("1. Start the web interface: cd ../python-backend && python main.py")
        print("2. Open http://localhost:8000 in your browser")
        print("3. Click 'Register Face' and follow the instructions")
        print("4. Then run this security monitor again")
        return
    
    print(f"\nFound {len(monitor.face_database)} registered faces:")
    for name in monitor.face_database.keys():
        print(f"  - {name}")
    
    print(f"\nSecurity Settings:")
    print(f"  - Lock delay: {monitor.lock_delay}s (locks computer when you're away)")
    print(f"  - Unlock delay: {monitor.unlock_delay}s (unlocks when you return)")
    print(f"  - Startup delay: {monitor.startup_delay}s (waits before monitoring starts)")
    print(f"  - Similarity threshold: {monitor.similarity_threshold:.0%} (face recognition accuracy)")
    print(f"  - Auto-login: {'ENABLED' if monitor.auto_login_enabled else 'DISABLED'}")
    
    # Auto-login disabled; no guidance needed
    
    print(f"\nStarting monitoring at 2 FPS (optimized for efficiency)...")
    print("Controls:")
    print("  'q' - Quit monitoring")
    print("  'r' - Reload face database")
    print("  'a' - Toggle auto-login (if configured)")
    
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
