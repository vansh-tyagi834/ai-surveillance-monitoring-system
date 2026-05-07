import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import requests

class FaceRecognitionMonitor:
    def __init__(self):
        # Camera setup
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face recognition setup
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}
        self.student_names = {}
        
        # Monitoring variables
        self.current_student = None
        self.suspicious_activities = []
        self.monitoring_start_time = time.time()
        
        # Detection thresholds
        self.CONFIDENCE_THRESHOLD = 100  # Lower = more strict
        self.NO_FACE_THRESHOLD = 60      # frames
        self.UNKNOWN_FACE_THRESHOLD = 30 # frames
        
        # Counters
        self.no_face_count = 0
        self.unknown_face_count = 0
        self.frame_count = 0
        
        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 8000
        
        # Firebase URL
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"
    
    def init_camera(self):
        """Initialize camera"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            for index in range(3):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"✅ Camera ready: Index {index}, Backend {backend}")
                        return True
                cap.release()
        
        print("❌ No camera found!")
        return False
    
    def load_registered_students(self):
        """Load all registered student faces"""
        if not os.path.exists('student_data'):
            print("❌ No student data directory found!")
            return False
        
        faces = []
        labels = []
        student_id_map = {}
        label_counter = 0
        
        print("📚 Loading registered students...")
        
        # Load all student registration files
        for filename in os.listdir('student_data'):
            if filename.endswith('_registration.json'):
                filepath = os.path.join('student_data', filename)
                
                try:
                    with open(filepath, 'r') as f:
                        student_data = json.load(f)
                    
                    student_id = student_data.get('student_id')
                    photo_paths = student_data.get('photo_paths', [])
                    
                    if student_id and photo_paths:
                        # Load face images for this student
                        student_faces_loaded = 0
                        
                        for photo_path in photo_paths:
                            if os.path.exists(photo_path):
                                # Load and preprocess image
                                img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    # Resize to standard size
                                    img_resized = cv2.resize(img, (100, 100))
                                    faces.append(img_resized)
                                    labels.append(label_counter)
                                    student_faces_loaded += 1
                        
                        if student_faces_loaded > 0:
                            student_id_map[label_counter] = student_id
                            self.student_names[label_counter] = student_id
                            label_counter += 1
                            print(f"✅ Loaded {student_faces_loaded} faces for {student_id}")
                        else:
                            print(f"⚠️ No valid faces found for {student_id}")
                    
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        if len(faces) > 0:
            # Train face recognizer
            print(f"🧠 Training face recognizer with {len(faces)} face samples...")
            self.face_recognizer.train(faces, np.array(labels))
            self.known_faces = student_id_map
            print(f"✅ Face recognition ready! Registered students: {len(student_id_map)}")
            
            # Show registered students
            for label, student_id in student_id_map.items():
                print(f"   🎓 {student_id} (Label: {label})")
            
            return True
        else:
            print("❌ No face data loaded!")
            return False
    
    def recognize_face(self, face_roi):
        """Recognize face using trained model"""
        if not self.known_faces:
            return None, 0
        
        try:
            # Preprocess face
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Predict
            label, confidence = self.face_recognizer.predict(face_resized)
            
            # Check if confidence is good enough
            if confidence < self.CONFIDENCE_THRESHOLD:
                student_id = self.known_faces.get(label)
                return student_id, confidence
            else:
                return None, confidence
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 0
    
    def detect_motion(self, current_frame):
        """Detect motion between frames"""
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
    
    def log_suspicious_activity(self, activity_type, details=""):
        """Log suspicious activity"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        session_time = time.time() - self.monitoring_start_time
        
        activity = {
            'type': activity_type,
            'details': details,
            'timestamp': timestamp,
            'session_time': f"{session_time:.1f}s",
            'current_student': self.current_student
        }
        
        self.suspicious_activities.append(activity)
        
        alert_msg = f"🚨 {activity_type.upper()} - {timestamp}"
        if details:
            alert_msg += f" ({details})"
        
        print(alert_msg)
        
        # Send to Firebase (optional)
        try:
            alert_url = f"{self.firebase_url}/monitoring_alerts.json"
            requests.post(alert_url, json=activity, timeout=5)
        except:
            pass  # Silent fail for Firebase
    
    def start_monitoring(self):
        """Start face recognition monitoring"""
        print("🎯 FACE RECOGNITION MONITORING SYSTEM")
        print("=" * 60)
        
        # Initialize camera
        if not self.init_camera():
            return
        
        # Load registered students
        if not self.load_registered_students():
            print("❌ Cannot start monitoring without registered students!")
            return
        
        print(f"\n🔍 Monitoring started at {datetime.now().strftime('%H:%M:%S')}")
        print("📋 Press 'q' to quit, 's' to save report")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )
            
            # Motion detection
            motion_detected = self.detect_motion(frame)
            
            # Display header info
            cv2.putText(frame, f"Monitoring Time: {int(time.time() - self.monitoring_start_time)}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Alerts: {len(self.suspicious_activities)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Motion: {'YES' if motion_detected else 'NO'}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if motion_detected else (0, 255, 0), 2)
            
            # Current student display
            if self.current_student:
                cv2.putText(frame, f"Student: {self.current_student}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Process faces
            if len(faces) == 0:
                # No face detected
                self.no_face_count += 1
                cv2.putText(frame, f"No Face Count: {self.no_face_count}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                if self.no_face_count > self.NO_FACE_THRESHOLD:
                    cv2.putText(frame, "⚠️ STUDENT MISSING!", 
                               (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if self.no_face_count == self.NO_FACE_THRESHOLD + 1:  # Log once
                        self.log_suspicious_activity("Student Missing", f"No face for {self.NO_FACE_THRESHOLD} frames")
            
            elif len(faces) == 1:
                # Single face - process recognition
                self.no_face_count = 0
                x, y, w, h = faces[0]
                
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face
                student_id, confidence = self.recognize_face(face_roi)
                
                if student_id:
                    # Known student recognized
                    self.unknown_face_count = 0
                    if self.current_student != student_id:
                        self.current_student = student_id
                        print(f"👤 Student identified: {student_id} (Confidence: {confidence:.1f})")
                    
                    # Draw green rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_id}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Eye detection
                    eyes = self.eye_cascade.detectMultiScale(face_roi)
                    if len(eyes) >= 2:
                        cv2.putText(frame, "👁️ Eyes OK", (x+w+10, y+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, "👁️ Eyes?", (x+w+10, y+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                else:
                    # Unknown face
                    self.unknown_face_count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    
                    if self.unknown_face_count > self.UNKNOWN_FACE_THRESHOLD:
                        cv2.putText(frame, "⚠️ UNREGISTERED PERSON!", 
                                   (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if self.unknown_face_count == self.UNKNOWN_FACE_THRESHOLD + 1:  # Log once
                            self.log_suspicious_activity("Unknown Person", f"Unregistered face detected")
            
            else:
                # Multiple faces
                self.no_face_count = 0
                cv2.putText(frame, "⚠️ MULTIPLE PEOPLE DETECTED!", 
                           (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Log suspicious activity
                if self.frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                    self.log_suspicious_activity("Multiple People", f"{len(faces)} faces detected")
            
            # Display frame
            cv2.imshow("🧠 AI Proctoring - Face Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_monitoring_report()
        
        # Cleanup
        self.cleanup()
    
    def save_monitoring_report(self):
        """Save monitoring report"""
        session_duration = time.time() - self.monitoring_start_time
        
        report = {
            'session_start': datetime.fromtimestamp(self.monitoring_start_time).isoformat(),
            'session_duration_seconds': session_duration,
            'session_duration_formatted': f"{int(session_duration//60)}m {int(session_duration%60)}s",
            'total_alerts': len(self.suspicious_activities),
            'monitored_student': self.current_student,
            'total_frames_processed': self.frame_count,
            'suspicious_activities': self.suspicious_activities,
            'report_generated': datetime.now().isoformat()
        }
        
        filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Monitoring report saved: {filename}")
        print(f"📊 Session duration: {report['session_duration_formatted']}")
        print(f"🚨 Total alerts: {len(self.suspicious_activities)}")
        
        # Send to Firebase
        try:
            report_url = f"{self.firebase_url}/monitoring_reports.json"
            requests.post(report_url, json=report, timeout=10)
            print("🔥 Report uploaded to Firebase!")
        except Exception as e:
            print(f"⚠️ Firebase upload failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save report
        self.save_monitoring_report()
        print("🎯 Monitoring system shutdown complete")

# 🚀 Run the monitoring system
if __name__ == "__main__":
    try:
        monitor = FaceRecognitionMonitor()
        monitor.start_monitoring()
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()