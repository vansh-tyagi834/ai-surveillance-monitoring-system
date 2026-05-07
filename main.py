import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import requests

class IntegratedProctoringSystem:
    def __init__(self):
        # Firebase
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"

        # Camera
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Face recognition
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}  # label -> student_id
        self.student_database = {}

        # System state
        self.current_student = None
        self.current_student_info = None
        self.exam_session_active = False
        self.session_start_time = 0
        self.suspicious_activities = []

        # Counters
        self.no_face_count = 0
        self.unknown_face_count = 0
        self.multiple_face_count = 0
        self.head_down_count = 0
        self.looking_side_count = 0

        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 8000

        # Head pose
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        self.camera_matrix = np.array([
            [640.0, 0.0, 320.0],
            [0.0, 640.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # RFID
        self.last_scan_count = 0
        self.last_rfid_check = 0
        self.rfid_check_interval = 2.0

        # Init
        print("🚀 INITIALIZING SYSTEM...")
        self.init_camera()
        self.create_directories()
        self.load_student_database()

    def create_directories(self):
        for d in ['student_faces', 'student_data', 'exam_reports']:
            if not os.path.exists(d):
                os.makedirs(d)

    def init_camera(self):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                time.sleep(2)
                print("✅ Camera ready")
                return
        print("❌ Cannot open camera!")
        exit()

    def load_student_database(self):
        faces = []
        labels = []
        label_id = 0
        self.known_faces.clear()

        if not os.path.exists('student_data'):
            return

        for file in os.listdir('student_data'):
            if file.endswith('_registration.json'):
                with open(f'student_data/{file}', 'r') as f:
                    data = json.load(f)
                student_id = data['student_id']
                self.student_database[student_id] = {
                    'name': data['name'],
                    'roll_number': data['roll_number'],
                    'photo_paths': data['photo_paths']
                }

                for img_path in data['photo_paths']:
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_resized = cv2.resize(img, (100, 100))
                            faces.append(img_resized)
                            labels.append(label_id)
                self.known_faces[label_id] = student_id
                label_id += 1

        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            print(f"✅ Trained with {len(self.known_faces)} students")
        else:
            print("⚠️ No face data found")

    def check_rfid_scans(self):
        current_time = time.time()
        if current_time - self.last_rfid_check < self.rfid_check_interval:
            return None
        self.last_rfid_check = current_time

        try:
            response = requests.get(f"{self.firebase_url}/rfid_scans.json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > self.last_scan_count:
                    latest_key = list(data.keys())[-1]
                    scan = data[latest_key]
                    self.last_scan_count = len(data)
                    return scan
        except Exception as e:
            print(f"⚠️ Firebase error: {e}")
        return None

    def wait_for_rfid_scan(self):
        print("🔍 Waiting for RFID scan...")
        while True:
            scan = self.check_rfid_scans()
            if scan:
                card_id = scan.get('student_id')
                print(f"🚨 RFID SCAN DETECTED: {card_id}")
                return card_id
            time.sleep(1)

    def recognize_face(self, face_roi):
        try:
            # Ensure face is grayscale and resized
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_roi, (100, 100))

            label, confidence = self.face_recognizer.predict(face_resized)
            if confidence < 90:
                student_id = self.known_faces.get(label)
                return student_id, confidence
            else:
                return None, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 100

    def get_facial_landmarks(self, gray, face):
        x, y, w, h = face
        roi = gray[y:y+h, x:x+w]
        roi = cv2.equalizeHist(roi)
        eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 3, minSize=(20, 20))
        landmarks = []

        # Nose tip
        landmarks.append((x + w // 2, y + h // 3))
        # Chin
        landmarks.append((x + w // 2, y + h - h // 8))
        # Eyes
        eye_count = 0
        for (ex, ey, ew, eh) in eyes:
            if eye_count >= 2:
                break
            landmarks.append((x + ex + ew // 2, y + ey + eh // 2))
            eye_count += 1
        # Mouth
        if len(landmarks) >= 4:
            landmarks.append((x + w // 4, y + 2 * h // 3))
            landmarks.append((x + 3 * w // 4, y + 2 * h // 3))
        return landmarks if len(landmarks) == 6 else None

    def calculate_head_pose(self, image_points):
        try:
            image_points = np.array(image_points, dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix,
                self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch = angles[0] * 180 / np.pi
                yaw = angles[1] * 180 / np.pi
                return pitch, yaw
        except:
            pass
        return None, None

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion = np.sum(thresh) > self.motion_threshold
        self.prev_frame = gray
        return motion

    def log_suspicious_activity(self, activity_type, details=""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity = {
            'type': activity_type,
            'details': details,
            'timestamp': timestamp,
            'student_id': self.current_student
        }
        self.suspicious_activities.append(activity)
        alert_msg = f"🚨 {activity_type.upper()}"
        if details:
            alert_msg += f" - {details}"
        print(f"{alert_msg} at {timestamp}")

    def start_monitoring_session(self):
        print(f"🎯 EXAM SESSION STARTED FOR {self.current_student}")
        self.exam_session_active = True
        self.session_start_time = time.time()
        self.suspicious_activities = []
        self.reset_counters()

        time.sleep(2)  # Warm-up

        while self.exam_session_active:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            # Status
            cv2.putText(frame, f"Session: {int(time.time() - self.session_start_time)}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Student: {self.current_student_info['name']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # No face
            if len(faces) == 0:
                self.no_face_count += 1
                if self.no_face_count == 30:
                    self.log_suspicious_activity("MISSING_STUDENT", "No face detected")
            else:
                self.no_face_count = 0

            # Multiple faces
            if len(faces) > 1:
                self.multiple_face_count += 1
                cv2.putText(frame, "⚠️ MULTIPLE FACES!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if self.multiple_face_count == 15:
                    self.log_suspicious_activity("MULTIPLE_FACES", f"{len(faces)} faces")
            else:
                self.multiple_face_count = 0

            # Single face
            recognized = False
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = frame_copy[y:y+h, x:x+w]

                student_id, confidence = self.recognize_face(face_roi)
                if student_id == self.current_student:
                    recognized = True
                    cv2.putText(frame, f"✅ {self.current_student_info['name']}",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Head pose
                    landmarks = self.get_facial_landmarks(gray, (x, y, w, h))
                    if landmarks and len(landmarks) == 6:
                        pitch, yaw = self.calculate_head_pose(landmarks)
                        if pitch is not None:
                            if pitch < -25:
                                self.head_down_count += 1
                                if self.head_down_count == 15:
                                    self.log_suspicious_activity("LOOKING_DOWN", f"Pitch: {pitch:.1f}°")
                            else:
                                self.head_down_count = 0

                            if abs(yaw) > 40:
                                self.looking_side_count += 1
                                if self.looking_side_count == 15:
                                    self.log_suspicious_activity("LOOKING_SIDE", f"Yaw: {yaw:.1f}°")
                            else:
                                self.looking_side_count = 0

                            cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    self.unknown_face_count += 1
                    if self.unknown_face_count == 30:
                        self.log_suspicious_activity("UNKNOWN_PERSON", f"Confidence: {confidence:.1f}")

            cv2.imshow("Proctoring Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.end_exam_session()

    def reset_counters(self):
        self.no_face_count = 0
        self.unknown_face_count = 0
        self.multiple_face_count = 0
        self.head_down_count = 0
        self.looking_side_count = 0

    def end_exam_session(self):
        self.save_exam_report()
        print("✅ Exam session ended")
        self.exam_session_active = False
        self.current_student = None

    def save_exam_report(self):
        duration = time.time() - self.session_start_time
        report = {
            "exam_session": {
                "student_id": self.current_student,
                "name": self.current_student_info.get('name', 'Unknown'),
                "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_sec": round(duration, 2),
                "total_alerts": len(self.suspicious_activities)
            },
            "alerts": self.suspicious_activities
        }
        filename = f"exam_reports/report_{self.current_student}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved: {filename}")

    def start_main_system(self):
        print("🎯 INTEGRATED PROCTORING SYSTEM")
        print("=" * 80)
        print(f"📊 Registered Students: {len(self.student_database)}")

        while True:
            card_id = self.wait_for_rfid_scan()
            if card_id in self.student_database:
                self.current_student = card_id
                self.current_student_info = self.student_database[card_id]
                print(f"✅ Welcome {self.current_student_info['name']}!")
                self.start_monitoring_session()
            else:
                print(f"❌ Student {card_id} not registered!")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        system = IntegratedProctoringSystem()
        system.start_main_system()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")