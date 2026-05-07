import cv2
import time
import json
from datetime import datetime

class ExamProctor:
    def __init__(self):
        # 🎯 Initialize camera with working config
        self.test_camera()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 📊 Monitoring variables
        self.no_face_count = 0
        self.multiple_face_count = 0
        self.suspicious_activity = []
        self.start_time = time.time()
        
        # ⚙️ Alert thresholds - INCREASED VALUES
        self.NO_FACE_THRESHOLD = 90  # frames (3 seconds at 30fps)
        self.MULTIPLE_FACE_THRESHOLD = 45
        
        # 🏃 Smart movement detection variables
        self.prev_frame = None
        self.motion_threshold = 8000       # Moderate sensitivity
        self.movement_history = []         # Track movement patterns
        self.head_positions = []           # Track head position changes
        self.suspicious_movement_count = 0
        self.SUSPICIOUS_MOVEMENT_THRESHOLD = 30
        
    def test_camera(self):
        """Find working camera configuration"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            for index in range(3):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"✅ Camera ready: Index {index}, Backend {backend}")
                        return
                cap.release()
        
        raise Exception("❌ No camera found!")
    
    def analyze_suspicious_movement(self, faces, frame, motion_detected):
        """Analyze movement patterns for cheating detection"""
        current_time = time.time()
        
        # Track head position changes
        if len(faces) == 1:
            x, y, w, h = faces[0]
            head_center = (x + w//2, y + h//2)
            
            # Store head positions with timestamp
            self.head_positions.append({
                'position': head_center,
                'timestamp': current_time
            })
            
            # Keep only last 5 seconds of data
            self.head_positions = [pos for pos in self.head_positions 
                                 if current_time - pos['timestamp'] < 5.0]
            
            # Analyze movement patterns
            if len(self.head_positions) > 10:
                self.detect_cheating_patterns(frame)
        
        # Store movement data
        self.movement_history.append({
            'motion': motion_detected,
            'timestamp': current_time,
            'face_count': len(faces)
        })
        
        # Keep only last 10 seconds
        self.movement_history = [m for m in self.movement_history 
                               if current_time - m['timestamp'] < 10.0]
    
    def detect_cheating_patterns(self, frame):
        """Detect specific cheating behaviors"""
        if len(self.head_positions) < 10:
            return
        
        current_time = time.time()
        recent_positions = [pos for pos in self.head_positions 
                          if current_time - pos['timestamp'] < 3.0]
        
        if len(recent_positions) < 5:
            return
        
        # Calculate movement metrics
        positions = [pos['position'] for pos in recent_positions]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Check for suspicious patterns
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Pattern 1: Excessive left-right movement (looking around)
        if x_range > 100:  # pixels
            cv2.putText(frame, "⚠️ LOOKING AROUND DETECTED", (10, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            self.log_suspicious_behavior("Looking around/side glances")
        
        # Pattern 2: Excessive up-down movement (looking up/down)
        if y_range > 80:
            cv2.putText(frame, "⚠️ HEAD MOVEMENT UP/DOWN", (10, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            self.log_suspicious_behavior("Excessive head up/down movement")
        
        # Pattern 3: Rapid movements (jittery behavior)
        if len(recent_positions) > 8 and (x_range > 60 or y_range > 50):
            cv2.putText(frame, "⚠️ RESTLESS BEHAVIOR", (10, 380), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 165), 2)
    def detect_motion(self, current_frame):
        """Enhanced motion detection"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray_current, gray_prev)
        blur = cv2.GaussianBlur(diff, (5,5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        
        # Count motion pixels
        motion_area = cv2.countNonZero(thresh)
        
        # Update previous frame
        self.prev_frame = current_frame.copy()
        
        return motion_area > self.motion_threshold
    
    def log_suspicious_behavior(self, behavior_type):
        """Log suspicious behavior (avoid duplicates)"""
        current_timestamp = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        alert = f"🚨 {behavior_type.upper()} - {current_time}"
        
        # Simple duplicate check - avoid same behavior in last 5 seconds
        if hasattr(self, 'last_behavior_time'):
            if (behavior_type in getattr(self, 'last_behavior_type', '') and 
                current_timestamp - getattr(self, 'last_behavior_time', 0) < 5.0):
                return  # Skip duplicate
        
        # Log the behavior
        self.suspicious_activity.append(alert)
        print(alert)
        
        # Update last behavior tracking
        self.last_behavior_type = behavior_type
        self.last_behavior_time = current_timestamp
    
    def detect_suspicious_activity(self, faces, frame):
        """Detect cheating patterns"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # 🚨 No face detected - FIXED DUPLICATE ALERTS
        if len(faces) == 0:
            self.no_face_count += 1
            cv2.putText(frame, f"No face count: {self.no_face_count}", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            if self.no_face_count > self.NO_FACE_THRESHOLD:
                alert = f"⚠️ STUDENT MISSING - {current_time}"
                if alert not in self.suspicious_activity:  # Avoid duplicates
                    self.suspicious_activity.append(alert)
                    print(alert)
                cv2.putText(frame, "STUDENT MISSING!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.no_face_count = 0
        
        # 🚨 Multiple faces detected - FIXED DUPLICATES
        if len(faces) > 1:
            self.multiple_face_count += 1
            if self.multiple_face_count == self.MULTIPLE_FACE_THRESHOLD:  # ONLY ONCE
                alert = f"🚨 MULTIPLE PEOPLE - {current_time}"
                self.suspicious_activity.append(alert)
                print(alert)
            cv2.putText(frame, "MULTIPLE PEOPLE DETECTED!", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.multiple_face_count = max(0, self.multiple_face_count - 1)  # GRADUAL DECREASE
    
    def detect_gaze(self, face, gray_frame, frame):
        """Basic gaze detection using eyes"""
        x, y, w, h = face
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # 👁️ Detect eyes
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            cv2.putText(frame, "👁️ Eyes detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(eyes) == 1:
            cv2.putText(frame, "⚠️ Looking away?", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "❌ No eyes", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw eye rectangles
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    def start_monitoring(self):
        """Main monitoring loop"""
        print("🎯 EXAM MONITORING STARTED")
        print("Press 'q' to quit, 's' to save report")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 🔍 Better face detection parameters
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,      # More sensitive
                minNeighbors=3,        # Less strict
                minSize=(50, 50),      # Smaller minimum size
                maxSize=(300, 300)     # Reasonable maximum size
            )
            
            # Debug info - REMOVE AFTER TESTING
            # print(f"Faces detected: {len(faces)}")
            
            # 🏃 Smart movement analysis
            motion_detected = self.detect_motion(frame)
            self.analyze_suspicious_movement(faces, frame, motion_detected)
            
            # Display motion status
            cv2.putText(frame, f"Motion: {'YES' if motion_detected else 'NO'}", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if motion_detected else (0, 255, 0), 2)
            
            # 📊 Status display
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Alerts: {len(self.suspicious_activity)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Motion: {'YES' if motion_detected else 'NO'}", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if motion_detected else (0, 255, 0), 2)
            
            # 🚨 Check for suspicious activity
            self.detect_suspicious_activity(faces, frame)
            
            # 👤 Process each face
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.detect_gaze(face, gray, frame)
            
            # 🖥️ Display
            cv2.imshow("🧠 AI Proctoring System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_report()
        
        self.cleanup()
    
    def save_report(self):
        """Save monitoring report"""
        report = {
            "session_duration": time.time() - self.start_time,
            "total_alerts": len(self.suspicious_activity),
            "alerts": self.suspicious_activity,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("📄 Report saved!")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("🎯 Monitoring ended")

# 🚀 Run the proctoring system
if __name__ == "__main__":
    try:
        proctor = ExamProctor()
        proctor.start_monitoring()
    except Exception as e:
        print(f"Error: {e}")