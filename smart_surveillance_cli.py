#!/usr/bin/env python3
"""
🎯 Smart IoT Surveillance System - FINAL COMPLETE VERSION
Fast, Secure RFID + Face Recognition + Proctored Exam Monitoring
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import threading
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import pyaudio
import audioop

class SmartSurveillanceSystem:
    def __init__(self):
        self.project_name = "🎯 Smart IoT Surveillance System"
        self.version = "v2.0 FINAL"
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"
        
        # Pre-initialize components for speed
        self.cap = None
        self.face_cascade = None
        self.eye_cascade = None
        self.face_recognizer = None
        self.audio = None
        
        # System status
        self.camera_ready = False
        self.firebase_connected = False
        self.face_model_trained = False
        self.registered_students = 0
        
        # Performance settings
        self.face_confidence_threshold = 80  # Stricter verification
        self.verification_required_frames = 45  # More frames for better verification
        
        # Quick setup and initialization
        self.quick_setup()
    
    def quick_setup(self):
        """Fast system initialization"""
        print("🚀 Quick System Setup...")
        
        # Create directories
        for directory in ['student_faces', 'student_data', 'exam_reports', 'logs', 'models']:
            os.makedirs(directory, exist_ok=True)
        
        # Pre-initialize camera
        self.init_camera_fast()
        
        # Pre-load face detection models
        self.load_face_models()
        
        # Check Firebase quickly
        self.check_firebase_fast()
        
        # Count registered students
        self.count_registered_students()
        
        # Initialize audio for sound detection
        self.init_audio()
        
        print("✅ Setup complete!")
    
    def init_camera_fast(self):
        """Fast camera initialization"""
        try:
            # Try common camera indices quickly
            for index in [0, 1]:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows optimized
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.cap = cap
                        self.camera_ready = True
                        print(f"📹 Camera ready: Index {index}")
                        return
                cap.release()
        except:
            pass
        print("❌ Camera not available")
    
    def load_face_models(self):
        """Pre-load face detection models"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Try to load existing model
            model_path = 'models/face_recognition_model.yml'
            if os.path.exists(model_path):
                self.face_recognizer.read(model_path)
                self.face_model_trained = True
                print("🧠 Face model loaded")
            
        except Exception as e:
            print(f"⚠️ Face models error: {e}")
    
    def check_firebase_fast(self):
        """Quick Firebase check"""
        try:
            response = requests.get(f"{self.firebase_url}/test.json", timeout=3)
            if response.status_code in [200, 404]:
                self.firebase_connected = True
        except:
            pass
    
    def count_registered_students(self):
        """Count registered students"""
        try:
            if os.path.exists('student_data'):
                files = [f for f in os.listdir('student_data') if f.endswith('_registration.json')]
                self.registered_students = len(files)
        except:
            self.registered_students = 0
    
    def init_audio(self):
        """Initialize audio for sound detection"""
        try:
            self.audio = pyaudio.PyAudio()
            print("🔊 Audio ready")
        except:
            print("⚠️ Audio not available")
    
    def display_status(self):
        """Fast status display"""
        print("\n" + "="*80)
        print(f"🎯 {self.project_name} {self.version}")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Camera {'✅' if self.camera_ready else '❌'} | Firebase {'✅' if self.firebase_connected else '❌'} | Students: {self.registered_students}")
        print("="*80)
    
    def main_menu(self):
        """Main menu"""
        print("\n🎯 CHOOSE OPTION:")
        print("1️⃣  RFID Registration + Face Capture")
        print("2️⃣  Train Face Recognition Model") 
        print("3️⃣  Start Exam Monitoring")
        print("4️⃣  View Reports & Students")
        print("5️⃣  Quick System Test")
        print("0️⃣  Exit")
        print("="*50)
    
    def wait_for_rfid_real_time(self):
        """Fast real-time RFID scanning"""
        print("🔍 Scanning for RFID... (Ctrl+C to cancel)")
        
        # Get baseline scan count
        try:
            response = requests.get(f"{self.firebase_url}/rfid_scans.json", timeout=5)
            baseline_count = len(response.json()) if response.status_code == 200 and response.json() else 0
        except:
            baseline_count = 0
        
        scan_attempts = 0
        max_attempts = 30  # 1 minute timeout
        
        while scan_attempts < max_attempts:
            try:
                response = requests.get(f"{self.firebase_url}/rfid_scans.json", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    current_count = len(data) if data else 0
                    
                    if current_count > baseline_count:
                        # New scan detected
                        latest_scan = list(data.values())[-1]
                        student_id = latest_scan.get('student_id')
                        
                        print(f"\n🚨 RFID DETECTED: {student_id}")
                        return student_id
                
                print(f"⏳ Waiting... ({scan_attempts}/{max_attempts})", end="\r")
                time.sleep(2)
                scan_attempts += 1
                
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None
            except:
                time.sleep(1)
                scan_attempts += 1
        
        print("\n⏰ Timeout - No RFID detected")
        manual_id = input("Enter Student ID manually (or Enter to cancel): ").strip()
        return manual_id if manual_id else None
    
    def advanced_face_verification(self, student_id):
        """Advanced face verification with proper recognition"""
        print(f"\n👤 IDENTITY VERIFICATION: {student_id}")
        
        # Load student's registered photos
        json_file = f"student_data/{student_id}_registration.json"
        if not os.path.exists(json_file):
            print("❌ No registration found!")
            return False
        
        with open(json_file, 'r') as f:
            student_data = json.load(f)
        
        photo_paths = student_data.get('photo_paths', [])
        if not photo_paths:
            print("❌ No photos found!")
            return False
        
        # Load registered face samples
        registered_faces = []
        for photo_path in photo_paths:
            if os.path.exists(photo_path):
                img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (100, 100))
                    registered_faces.append(img_resized)
        
        if not registered_faces:
            print("❌ Could not load registered photos!")
            return False
        
        print("🔍 Look at camera for verification...")
        print("📊 Analyzing face features...")
        
        verification_scores = []
        frames_processed = 0
        required_frames = 30
        
        while frames_processed < required_frames:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            # Display progress
            progress = int((frames_processed / required_frames) * 100)
            cv2.putText(frame, f"Verifying {student_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {progress}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and compare face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Compare with registered faces using template matching
                max_similarity = 0
                for registered_face in registered_faces:
                    # Calculate similarity using normalized cross-correlation
                    result = cv2.matchTemplate(face_resized, registered_face, cv2.TM_CCOEFF_NORMED)
                    similarity = np.max(result) * 100
                    max_similarity = max(max_similarity, similarity)
                
                verification_scores.append(max_similarity)
                frames_processed += 1
                
                # Show real-time similarity
                cv2.putText(frame, f"Similarity: {max_similarity:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Position face properly", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("🔍 Identity Verification", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Calculate verification result
        if verification_scores:
            avg_score = np.mean(verification_scores)
            max_score = np.max(verification_scores)
            
            print(f"📊 Verification Results:")
            print(f"   Average Similarity: {avg_score:.1f}%")
            print(f"   Maximum Similarity: {max_score:.1f}%")
            print(f"   Frames Analyzed: {len(verification_scores)}")
            
            # Strict verification criteria
            if avg_score >= 75 and max_score >= 85:
                print("✅ IDENTITY VERIFIED!")
                return True
            else:
                print("❌ IDENTITY VERIFICATION FAILED!")
                print("⚠️  Possible impersonation detected!")
                
                # Log security incident
                incident = {
                    'type': 'identity_verification_failed',
                    'student_id': student_id,
                    'avg_similarity': float(avg_score),
                    'max_similarity': float(max_score),
                    'timestamp': datetime.now().isoformat(),
                    'risk_level': 'HIGH'
                }
                
                with open(f"logs/security_incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                    json.dump(incident, f, indent=2)
                
                return False
        else:
            print("❌ No face data captured for verification!")
            return False
    
    def fast_photo_capture(self, student_id):
        """Fast photo capture with better UX"""
        print(f"\n📸 PHOTO CAPTURE: {student_id}")
        print("Position face → Press SPACE → 3 photos needed")
        
        captured = 0
        target = 3
        photo_paths = []
        
        while captured < target:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
            
            # UI
            cv2.putText(frame, f"{student_id} - Photo {captured+1}/{target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "READY - Press SPACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Capture
                    face_roi = frame[y:y+h, x:x+w]
                    filename = f"student_faces/{student_id}_photo_{captured+1}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, face_roi)
                    photo_paths.append(filename)
                    captured += 1
                    print(f"✅ Photo {captured} captured")
                    time.sleep(0.5)
                    
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return False, []
            else:
                cv2.putText(frame, "Show ONE face only", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"📸 {student_id}", frame)
        
        cv2.destroyAllWindows()
        
        # Save registration
        student_data = {
            'student_id': student_id,
            'registration_time': datetime.now().isoformat(),
            'photos_captured': captured,
            'photo_paths': photo_paths,
            'status': 'registered'
        }
        
        with open(f"student_data/{student_id}_registration.json", 'w') as f:
            json.dump(student_data, f, indent=2)
        
        print(f"✅ {student_id} registered successfully!")
        return True, photo_paths
    
    def check_duplicate_faces(self):
        """Check for duplicate faces across different student IDs"""
        print("\n🔍 Checking for duplicate faces...")
        
        student_files = [f for f in os.listdir('student_data') if f.endswith('_registration.json')]
        if len(student_files) < 2:
            print("📊 Not enough students to check duplicates")
            return
        
        # Load all student faces
        all_faces = {}
        for file in student_files:
            student_id = file.replace('_registration.json', '')
            with open(f"student_data/{file}", 'r') as f:
                data = json.load(f)
            
            photo_paths = data.get('photo_paths', [])
            faces = []
            for photo_path in photo_paths:
                if os.path.exists(photo_path):
                    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(cv2.resize(img, (100, 100)))
            
            if faces:
                all_faces[student_id] = faces
        
        # Compare faces between students
        print("🔍 Analyzing face similarities...")
        duplicates_found = []
        
        student_ids = list(all_faces.keys())
        for i in range(len(student_ids)):
            for j in range(i+1, len(student_ids)):
                student1, student2 = student_ids[i], student_ids[j]
                
                max_similarity = 0
                for face1 in all_faces[student1]:
                    for face2 in all_faces[student2]:
                        result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
                        similarity = np.max(result) * 100
                        max_similarity = max(max_similarity, similarity)
                
                if max_similarity > 80:  # High similarity threshold
                    duplicates_found.append((student1, student2, max_similarity))
                    print(f"🚨 POTENTIAL DUPLICATE: {student1} ↔ {student2} ({max_similarity:.1f}% similar)")
        
        if duplicates_found:
            print(f"\n⚠️  {len(duplicates_found)} potential duplicates found!")
            for student1, student2, similarity in duplicates_found:
                print(f"   🔴 {student1} ↔ {student2}: {similarity:.1f}% match")
        else:
            print("✅ No duplicate faces found")
    
    def train_face_model(self):
        """Train face recognition model"""
        print("\n🧠 TRAINING FACE RECOGNITION MODEL")
        
        # Collect all training data
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        for filename in os.listdir('student_data'):
            if filename.endswith('_registration.json'):
                student_id = filename.replace('_registration.json', '')
                
                with open(f"student_data/{filename}", 'r') as f:
                    data = json.load(f)
                
                photo_paths = data.get('photo_paths', [])
                student_faces = 0
                
                for photo_path in photo_paths:
                    if os.path.exists(photo_path):
                        img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_resized = cv2.resize(img, (100, 100))
                            faces.append(img_resized)
                            labels.append(current_label)
                            student_faces += 1
                
                if student_faces > 0:
                    label_map[current_label] = student_id
                    current_label += 1
                    print(f"✅ Loaded {student_faces} faces for {student_id}")
        
        if len(faces) > 0:
            print(f"🧠 Training with {len(faces)} face samples...")
            self.face_recognizer.train(faces, np.array(labels))
            
            # Save model and label map
            os.makedirs('models', exist_ok=True)
            self.face_recognizer.save('models/face_recognition_model.yml')
            
            with open('models/label_map.json', 'w') as f:
                json.dump(label_map, f, indent=2)
            
            self.face_model_trained = True
            print("✅ Face recognition model trained and saved!")
        else:
            print("❌ No training data found!")
    
    def start_exam_monitoring(self):
        """Start comprehensive exam monitoring"""
        if not self.camera_ready:
            print("❌ Camera not available!")
            return
        
        if not self.face_model_trained:
            print("❌ Face model not trained! Train model first.")
            return
        
        print("\n🎯 STARTING EXAM MONITORING")
        print("Features: Face Recognition + Motion + Sound Detection")
        
        # Load face model and label map
        try:
            with open('models/label_map.json', 'r') as f:
                label_map = json.load(f)
            # Convert string keys to int
            label_map = {int(k): v for k, v in label_map.items()}
        except:
            print("❌ Could not load label map!")
            return
        
        # Monitoring variables
        monitoring_start = time.time()
        alerts = []
        current_student = None
        no_face_count = 0
        unknown_face_count = 0
        prev_frame = None
        
        # Audio monitoring setup
        sound_alerts = []
        
        def monitor_audio():
            """Background audio monitoring"""
            if not self.audio:
                return
            
            try:
                stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
                
                while True:
                    data = stream.read(1024, exception_on_overflow=False)
                    volume = audioop.rms(data, 2)
                    
                    if volume > 1000:  # Adjust threshold as needed
                        sound_alerts.append({
                            'type': 'loud_sound',
                            'volume': volume,
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"🔊 LOUD SOUND DETECTED: {volume}")
                    
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Audio monitoring error: {e}")
        
        # Start audio monitoring in background
        if self.audio:
            audio_thread = threading.Thread(target=monitor_audio, daemon=True)
            audio_thread.start()
        
        print("🎯 Monitoring active! Press 'q' to stop, 's' to save report")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
            
            # Motion detection
            motion_detected = False
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_area = cv2.countNonZero(thresh)
                motion_detected = motion_area > 5000
            prev_frame = gray.copy()
            
            # Display header
            session_time = int(time.time() - monitoring_start)
            cv2.putText(frame, f"Monitoring: {session_time}s | Alerts: {len(alerts)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(faces)} | Motion: {'YES' if motion_detected else 'NO'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if current_student:
                cv2.putText(frame, f"Student: {current_student}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Process faces
            if len(faces) == 0:
                no_face_count += 1
                cv2.putText(frame, f"No Face: {no_face_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if no_face_count > 60:  # 2 seconds
                    cv2.putText(frame, "⚠️ STUDENT MISSING!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if no_face_count == 61:  # Log once
                        alerts.append({
                            'type': 'student_missing',
                            'timestamp': datetime.now().isoformat(),
                            'session_time': session_time
                        })
            
            elif len(faces) == 1:
                no_face_count = 0
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Face recognition
                try:
                    label, confidence = self.face_recognizer.predict(face_resized)
                    
                    if confidence < self.face_confidence_threshold and label in label_map:
                        # Known student
                        student_id = label_map[label]
                        if current_student != student_id:
                            current_student = student_id
                            print(f"👤 Student recognized: {student_id}")
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{student_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        unknown_face_count = 0
                    else:
                        # Unknown face
                        unknown_face_count += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if unknown_face_count > 30:  # 1 second
                            cv2.putText(frame, "⚠️ UNAUTHORIZED PERSON!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if unknown_face_count == 31:
                                alerts.append({
                                    'type': 'unauthorized_person',
                                    'timestamp': datetime.now().isoformat(),
                                    'session_time': session_time
                                })
                
                except:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "PROCESSING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            else:
                # Multiple faces
                cv2.putText(frame, "⚠️ MULTIPLE PEOPLE!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Log alert
                if session_time % 5 == 0:  # Every 5 seconds
                    alerts.append({
                        'type': 'multiple_people',
                        'count': len(faces),
                        'timestamp': datetime.now().isoformat(),
                        'session_time': session_time
                    })
            
            # Show sound alerts
            if sound_alerts and len(sound_alerts) > len([a for a in alerts if a.get('type') == 'loud_sound']):
                cv2.putText(frame, "🔊 LOUD SOUND!", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                alerts.extend([a for a in sound_alerts if a not in alerts])
            
            cv2.imshow("🎯 Exam Monitoring", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_monitoring_report(alerts, sound_alerts, monitoring_start, current_student)
        
        cv2.destroyAllWindows()
        self.save_monitoring_report(alerts, sound_alerts, monitoring_start, current_student)
        print("🎯 Monitoring stopped")
    
    def save_monitoring_report(self, alerts, sound_alerts, start_time, student):
        """Save comprehensive monitoring report"""
        duration = time.time() - start_time
        
        report = {
            'session_info': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'duration_seconds': duration,
                'duration_formatted': f"{int(duration//60)}m {int(duration%60)}s",
                'monitored_student': student
            },
            'alerts_summary': {
                'total_alerts': len(alerts),
                'sound_alerts': len(sound_alerts),
                'alert_types': {}
            },
            'detailed_alerts': alerts,
            'sound_alerts': sound_alerts,
            'report_generated': datetime.now().isoformat()
        }
        
        # Count alert types
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            report['alerts_summary']['alert_types'][alert_type] = report['alerts_summary']['alert_types'].get(alert_type, 0) + 1
        
        filename = f"exam_reports/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved: {filename}")
        print(f"📊 Session: {report['session_info']['duration_formatted']}")
        print(f"🚨 Total alerts: {len(alerts)}")
        
        # Upload to Firebase
        try:
            requests.post(f"{self.firebase_url}/monitoring_reports.json", json=report, timeout=5)
            print("🔥 Uploaded to Firebase!")
        except:
            print("⚠️ Firebase upload failed")
    
    def view_reports_and_students(self):
        """View comprehensive reports and student info"""
        print("\n📊 SYSTEM REPORTS & STUDENTS")
        print("="*60)
        
        # Show registered students
        print(f"👥 REGISTERED STUDENTS: {self.registered_students}")
        if self.registered_students > 0:
            for filename in os.listdir('student_data'):
                if filename.endswith('_registration.json'):
                    student_id = filename.replace('_registration.json', '')
                    try:
                        with open(f"student_data/{filename}", 'r') as f:
                            data = json.load(f)
                        reg_time = data.get('registration_time', 'Unknown')
                        photos = data.get('photos_captured', 0)
                        print(f"   📱 {student_id} - {photos} photos - {reg_time[:10]}")
                    except:
                        print(f"   📱 {student_id} - Error reading data")
        
        # Show exam reports
        print(f"\n📄 EXAM REPORTS:")
        if os.path.exists('exam_reports'):
            reports = [f for f in os.listdir('exam_reports') if f.endswith('.json')]
            if reports:
                print(f"   Total reports: {len(reports)}")
                for report in sorted(reports)[-5:]:  # Last 5 reports
                    try:
                        with open(f"exam_reports/{report}", 'r') as f:
                            data = json.load(f)
                        duration = data.get('session_info', {}).get('duration_formatted', 'Unknown')
                        alerts = data.get('alerts_summary', {}).get('total_alerts', 0)
                        student = data.get('session_info', {}).get('monitored_student', 'Unknown')
                        print(f"   📋 {report[:20]}... - {student} - {duration} - {alerts} alerts")
                    except:
                        print(f"   📋 {report} - Error reading")
            else:
                print("   No reports found")
        
        # Show security incidents
        print(f"\n🚨 SECURITY INCIDENTS:")
        if os.path.exists('logs'):
            incidents = [f for f in os.listdir('logs') if f.startswith('security_')]
            if incidents:
                print(f"   Total incidents: {len(incidents)}")
                for incident in sorted(incidents)[-3:]:  # Last 3 incidents
                    try:
                        with open(f"logs/{incident}", 'r') as f:
                            data = json.load(f)
                        incident_type = data.get('type', 'Unknown')
                        student = data.get('student_id', 'Unknown')
                        risk = data.get('risk_level', 'Unknown')
                        print(f"   🔴 {incident_type} - {student} - {risk} RISK")
                    except:
                        print(f"   🔴 {incident} - Error reading")
            else:
                print("   No security incidents")
        
        input("\nPress Enter to continue...")
    
    def quick_system_test(self):
        """Quick system component test"""
        print("\n🔧 QUICK SYSTEM TEST")
        print("="*40)
        
        # Test camera
        print("📹 Testing camera...")
        if self.camera_ready:
            print("   ✅ Camera OK")
            ret, frame = self.cap.read()
            if ret:
                print("   ✅ Frame capture OK")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray)
                print(f"   📊 Faces detected: {len(faces)}")
            else:
                print("   ❌ Frame capture failed")
        else:
            print("   ❌ Camera not available")
        
        # Test Firebase
        print("\n🔥 Testing Firebase...")
        if self.firebase_connected:
            print("   ✅ Firebase connected")
            try:
                # Test read/write
                test_data = {'test': time.time()}
                response = requests.post(f"{self.firebase_url}/test.json", json=test_data, timeout=5)
                if response.status_code == 200:
                    print("   ✅ Firebase read/write OK")
                else:
                    print("   ⚠️ Firebase write issue")
            except:
                print("   ⚠️ Firebase test failed")
        else:
            print("   ❌ Firebase not connected")
        
        # Test face model
        print("\n🧠 Testing face recognition...")
        if self.face_model_trained:
            print("   ✅ Face model loaded")
            if os.path.exists('models/label_map.json'):
                with open('models/label_map.json', 'r') as f:
                    label_map = json.load(f)
                print(f"   📊 Trained students: {len(label_map)}")
            else:
                print("   ⚠️ Label map missing")
        else:
            print("   ❌ Face model not trained")
        
        # Test audio
        print("\n🔊 Testing audio...")
        if self.audio:
            print("   ✅ Audio system ready")
            try:
                # Quick audio test
                stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
                data = stream.read(1024, exception_on_overflow=False)
                volume = audioop.rms(data, 2)
                print(f"   📊 Current audio level: {volume}")
                stream.close()
            except:
                print("   ⚠️ Audio test failed")
        else:
            print("   ❌ Audio not available")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop - FAST VERSION"""
        print("🚀 Smart IoT Surveillance System - FINAL VERSION")
        
        while True:
            try:
                self.display_status()
                self.main_menu()
                
                choice = input("🎯 Choose (0-5): ").strip()
                
                if choice == '1':
                    # RFID Registration + Face Capture
                    if not self.camera_ready:
                        print("❌ Camera required!")
                        continue
                    
                    print("\n🔍 RFID + REGISTRATION WORKFLOW")
                    student_id = self.wait_for_rfid_real_time()
                    
                    if student_id:
                        # Check if registered
                        json_file = f"student_data/{student_id}_registration.json"
                        if os.path.exists(json_file):
                            print(f"✅ {student_id} already registered!")
                            
                            choice = input("Verify identity? (y/n): ").strip().lower()
                            if choice == 'y':
                                if self.advanced_face_verification(student_id):
                                    print("✅ Identity verified! Ready for exam.")
                                else:
                                    print("❌ Identity verification FAILED!")
                        else:
                            # New registration
                            print(f"🆕 New student: {student_id}")
                            success, photos = self.fast_photo_capture(student_id)
                            if success:
                                self.registered_students += 1
                                print(f"✅ {student_id} registered successfully!")
                
                elif choice == '2':
                    # Train Face Model
                    self.check_duplicate_faces()
                    self.train_face_model()
                
                elif choice == '3':
                    # Start Monitoring
                    self.start_exam_monitoring()
                
                elif choice == '4':
                    # View Reports
                    self.view_reports_and_students()
                
                elif choice == '5':
                    # System Test
                    self.quick_system_test()
                
                elif choice == '0':
                    print("👋 Shutting down...")
                    break
                
                else:
                    print("❌ Invalid choice!")
                
            except KeyboardInterrupt:
                print("\n❌ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.audio:
            self.audio.terminate()
        cv2.destroyAllWindows()
        print("🎯 System shutdown complete!")

# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 Initializing Smart IoT Surveillance System...")
    
    # Check requirements
    if sys.version_info < (3, 6):
        print("❌ Python 3.6+ required!")
        sys.exit(1)
    
    try:
        # Quick dependency check
        import cv2
        import numpy as np
        import requests
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install: pip install opencv-python numpy requests pyaudio")
        sys.exit(1)
    
    # Initialize and run
    try:
        system = SmartSurveillanceSystem()
        system.run()
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()