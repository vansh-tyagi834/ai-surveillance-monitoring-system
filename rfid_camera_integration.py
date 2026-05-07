import requests
import json
import time
import cv2
import os
from datetime import datetime

class RFIDCameraSystem:
    def __init__(self):
        # Firebase URL
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com/rfid_scans.json"
        
        # Camera setup
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create directories
        self.create_directories()
        
        # Last scan count for monitoring
        self.last_scan_count = 0
        
    def create_directories(self):
        """Create directories for storing data"""
        directories = ['student_faces', 'student_data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
    
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
    
    def get_latest_rfid_scan(self):
        """Get latest RFID scan from Firebase"""
        try:
            response = requests.get(self.firebase_url)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > self.last_scan_count:
                    # Get latest scan
                    latest_key = list(data.keys())[-1]
                    latest_scan = data[latest_key]
                    latest_scan['firebase_key'] = latest_key
                    
                    self.last_scan_count = len(data)
                    return latest_scan
            
            return None
            
        except Exception as e:
            print(f"❌ Firebase error: {e}")
            return None
    
    def capture_student_photos(self, student_id):
        """Capture photos for student registration"""
        if not self.cap:
            print("❌ Camera not initialized!")
            return False
        
        print(f"\n📸 CAMERA ACTIVATED for Student: {student_id}")
        print("=" * 60)
        print("📋 Instructions:")
        print("1. Position your face in the green rectangle")
        print("2. Press SPACE to capture photo")
        print("3. Take 3 photos from different angles")
        print("4. Press Q to quit")
        print("=" * 60)
        
        captured_photos = 0
        target_photos = 3
        photo_paths = []
        
        while captured_photos < target_photos:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Display UI elements
            cv2.putText(frame, f"Student ID: {student_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Photos: {captured_photos}/{target_photos}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Process faces
            if len(faces) == 1:
                # Single face detected - good for capture
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, "READY TO CAPTURE!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif len(faces) > 1:
                # Multiple faces
                cv2.putText(frame, "Multiple faces! Show only one face", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
            else:
                # No face detected
                cv2.putText(frame, "No face detected! Position properly", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow(f"📸 Registration - {student_id}", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(faces) == 1:  # Space to capture
                # Extract face
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                # Save photo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_filename = f"student_faces/{student_id}_photo_{captured_photos + 1}_{timestamp}.jpg"
                cv2.imwrite(photo_filename, face_roi)
                photo_paths.append(photo_filename)
                
                captured_photos += 1
                print(f"✅ Photo {captured_photos} saved: {photo_filename}")
                
                # Brief pause
                time.sleep(1)
                
            elif key == ord('q'):  # Quit
                print("❌ Photo capture cancelled")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        
        # Save student data
        student_data = {
            'student_id': student_id,
            'registration_time': datetime.now().isoformat(),
            'photos_captured': captured_photos,
            'photo_paths': photo_paths,
            'status': 'registered'
        }
        
        # Save to JSON file
        json_filename = f"student_data/{student_id}_registration.json"
        with open(json_filename, 'w') as f:
            json.dump(student_data, f, indent=2)
        
        print(f"✅ Registration completed for {student_id}!")
        print(f"📄 Data saved: {json_filename}")
        print(f"📸 Photos saved: {len(photo_paths)}")
        
        return True
    
    def start_monitoring(self):
        """Start the main monitoring loop"""
        print("🎯 RFID + CAMERA SYSTEM STARTED")
        print("=" * 60)
        
        # Initialize camera
        if not self.init_camera():
            print("❌ Cannot proceed without camera!")
            return
        
        # Get initial scan count
        try:
            response = requests.get(self.firebase_url)
            if response.status_code == 200:
                data = response.json()
                self.last_scan_count = len(data) if data else 0
                print(f"📊 Starting with {self.last_scan_count} existing scans")
        except:
            print("⚠️ Could not get initial scan count")
        
        print("\n🔍 Waiting for RFID scans...")
        print("📡 Scan your RFID card on Arduino...")
        
        while True:
            try:
                # Check for new RFID scan
                latest_scan = self.get_latest_rfid_scan()
                
                if latest_scan:
                    student_id = latest_scan.get('student_id')
                    firebase_key = latest_scan.get('firebase_key')
                    
                    print(f"\n🚨 NEW RFID SCAN DETECTED!")
                    print(f"📱 Student ID: {student_id}")
                    print(f"🔑 Firebase Key: {firebase_key}")
                    
                    # Check if already registered
                    json_file = f"student_data/{student_id}_registration.json"
                    if os.path.exists(json_file):
                        print(f"⚠️ Student {student_id} already registered!")
                        choice = input("Re-register? (y/n): ").lower()
                        if choice != 'y':
                            continue
                    
                    # Start photo capture
                    print(f"🎯 Starting registration for {student_id}...")
                    success = self.capture_student_photos(student_id)
                    
                    if success:
                        print(f"✅ SUCCESS: {student_id} registered!")
                    else:
                        print(f"❌ FAILED: {student_id} registration failed!")
                    
                    # Ask to continue
                    choice = input("\nContinue monitoring? (y/n): ").lower()
                    if choice != 'y':
                        break
                
                # Wait before next check
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n❌ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(3)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🎯 System shutdown complete")

# 🚀 Run the system
if __name__ == "__main__":
    try:
        system = RFIDCameraSystem()
        system.start_monitoring()
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()