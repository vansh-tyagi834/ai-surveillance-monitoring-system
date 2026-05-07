
"""
🎯 STEP 2: CAMERA TRIGGER & FACE CAPTURE
=========================================

When Arduino attendance is detected, this will:
1. Initialize camera
2. Capture face photo
3. Save with student ID
4. Display preview

STANDALONE TEST FILE - Test this before integrating!
"""

import cv2
import requests
import time
import threading
from datetime import datetime
import os

class ArduinoAttendanceMonitor:
    """Monitor Firebase for Arduino attendance (from Step 1)"""

    def __init__(self, camera_handler=None):
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"
        self.last_record_id = None
        self.monitoring = False
        self.camera_handler = camera_handler  # NEW: Camera handler

        print("✅ Arduino Attendance Monitor initialized")

    def start_monitoring(self):
        """Start monitoring Firebase"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Started monitoring Arduino attendance...")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("⏸️ Stopped monitoring")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                new_record = self._check_new_attendance()

                if new_record:
                    self._handle_new_attendance(new_record)

                time.sleep(2)

            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(5)

    def _check_new_attendance(self):
        """Check Firebase for new records"""
        try:
            url = f"{self.firebase_url}/attendance_records.json"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if data:
                    latest_id = list(data.keys())[-1]

                    if latest_id != self.last_record_id:
                        self.last_record_id = latest_id
                        return {
                            'id': latest_id,
                            'data': data[latest_id]
                        }

            return None

        except Exception as e:
            print(f"⚠️ Error checking Firebase: {e}")
            return None

    def _handle_new_attendance(self, record):
        """Handle new attendance - TRIGGER CAMERA! (NEW!)"""
        record_id = record['id']
        data = record['data']

        student_id = data.get('student_id', 'Unknown')
        fingerprint_id = data.get('fingerprint_id', 0)
        confidence = data.get('confidence', 0)
        status = data.get('status', 'unknown')

        print("\n" + "="*60)
        print("🎯 NEW ARDUINO ATTENDANCE DETECTED!")
        print("="*60)
        print(f"📱 Card ID: {student_id}")
        print(f"👆 Fingerprint ID: {fingerprint_id}")
        print(f"📊 Confidence: {confidence}")
        print(f"✅ Status: {status}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # NEW: Trigger camera capture!
        if self.camera_handler and status == "success":
            print("\n📸 Triggering camera capture...")
            self.camera_handler.capture_face(student_id, record_id)

        return record


class CameraHandler:
    """Handle camera operations for face capture (NEW!)"""

    def __init__(self):
        self.cap = None
        self.camera_active = False

        # Create directory for captured images
        self.capture_dir = "attendance_captures"
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
            print(f"✅ Created directory: {self.capture_dir}")

        print("✅ Camera Handler initialized")

    def initialize_camera(self):
        """Initialize camera"""
        try:
            print("📸 Initializing camera...")
            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                print("❌ Failed to open camera!")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test read
            ret, frame = self.cap.read()
            if ret:
                self.camera_active = True
                print("✅ Camera initialized successfully!")
                return True
            else:
                print("❌ Failed to read from camera!")
                return False

        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False

    def capture_face(self, student_id, record_id):
        """Capture face photo when attendance is marked"""
        try:
            # Initialize camera if not active
            if not self.camera_active:
                if not self.initialize_camera():
                    print("❌ Cannot capture - camera not available")
                    return None

            print("\n" + "="*60)
            print("📸 CAPTURING FACE PHOTO")
            print("="*60)

            # Give user time to look at camera
            print("⏳ Preparing... (2 seconds)")
            time.sleep(2)

            # Capture frame
            ret, frame = self.cap.read()

            if not ret:
                print("❌ Failed to capture frame!")
                return None

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{student_id}_{timestamp}_{record_id[:8]}.jpg"
            filepath = os.path.join(self.capture_dir, filename)

            # Save image
            cv2.imwrite(filepath, frame)
            print(f"✅ Photo saved: {filename}")

            # Detect face in captured image
            face_detected = self._detect_face(frame)

            if face_detected:
                print("✅ Face detected in photo!")
            else:
                print("⚠️ No face detected - photo saved anyway")

            # Display preview (optional)
            print("\n👁️ Displaying preview... (Press any key to close)")
            cv2.imshow('Captured Photo', frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()

            print("="*60)
            print(f"📁 Saved to: {filepath}")
            print("="*60 + "\n")

            return filepath

        except Exception as e:
            print(f"❌ Capture error: {e}")
            return None

    def _detect_face(self, frame):
        """Simple face detection using Haar Cascade"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            return len(faces) > 0

        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            return False

    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.camera_active = False
            print("📸 Camera released")


# ============================================
# STEP 2 TESTING
# ============================================

if __name__ == "__main__":
    print("\n🎯 STEP 2: CAMERA TRIGGER & CAPTURE TEST")
    print("="*60)

    # Create camera handler
    print("\n1️⃣ Initializing camera handler...")
    camera = CameraHandler()

    # Test camera initialization
    print("\n2️⃣ Testing camera...")
    if camera.initialize_camera():
        print("✅ Camera test successful!")
    else:
        print("❌ Camera test failed!")
        print("\nTroubleshooting:")
        print("- Is webcam connected?")
        print("- Is webcam being used by another app?")
        print("- Try closing other camera apps")
        exit(1)

    # Create monitor with camera handler
    print("\n3️⃣ Starting Arduino attendance monitor...")
    monitor = ArduinoAttendanceMonitor(camera_handler=camera)
    monitor.start_monitoring()

    print("\n" + "="*60)
    print("📋 READY TO TEST!")
    print("="*60)
    print("\n📝 Instructions:")
    print("1. Make sure Arduino is running")
    print("2. Scan RFID card on Arduino")
    print("3. Place finger on fingerprint sensor")
    print("4. Arduino will send to Firebase")
    print("5. Python will detect and capture your face!")
    print("6. Photo will be saved in 'attendance_captures' folder")
    print("\n⏳ Monitoring... (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏸️ Stopping system...")
        monitor.stop_monitoring()
        camera.release_camera()
        print("✅ System stopped!")
        print("\n📁 Check 'attendance_captures' folder for saved photos!")
