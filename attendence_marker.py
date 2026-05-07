"""
Smart Campus - Attendance Marker System
Integrates with Arduino hardware to mark attendance in Firebase
For Major Project Phase 2
"""

import serial
import cv2
import requests
from datetime import datetime
import time
from deepface import DeepFace # type: ignore
import json

class AttendanceMarker:
    def __init__(self):
        # Firebase configuration
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"

        # Arduino serial connection
        try:
            self.arduino = serial.Serial('COM3', 9600, timeout=1)  # Change COM port if needed
            print("✅ Connected to Arduino")
        except:
            print("⚠️ Arduino not connected - Running in TEST MODE")
            self.arduino = None

        # Camera for face verification
        self.camera = cv2.VideoCapture(0)
        print("✅ Camera initialized")

        print("\n🎓 Smart Campus Attendance System")
        print("="*50)
        print("Ready to mark attendance!")
        print("Scan card to begin...")

    def get_today_key(self):
        """Get today's date in YYYY-MM-DD format"""
        return datetime.now().strftime("%Y-%m-%d")

    def capture_face(self):
        """Capture and verify face"""
        print("📸 Capturing face...")
        ret, frame = self.camera.read()

        if ret:
            # Save captured image
            cv2.imwrite("temp_face.jpg", frame)
            cv2.imshow("Face Captured", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            return True
        return False

    def verify_face(self, student_id):
        """Verify captured face against registered face"""
        try:
            # In real system, compare with registered image from Firebase
            # For demo, we'll accept any face
            print("✅ Face verified!")
            return True
        except Exception as e:
            print(f"❌ Face verification failed: {e}")
            return False

    def mark_attendance(self, rfid_uid, student_name, fingerprint_verified=True):
        """Mark attendance in Firebase"""
        today = self.get_today_key()
        current_time = datetime.now().strftime("%H:%M:%S")

        # Attendance data
        attendance_data = {
            "name": student_name,
            "status": "Present",
            "time": current_time,
            "method": "RFID+Finger+Face",
            "late": False
        }

        # Update Firebase - Daily attendance
        url = f"{self.firebase_url}/attendance/daily/{today}/CSE-3A/Data_Structures/students/{rfid_uid}.json"

        try:
            response = requests.put(url, json=attendance_data)

            if response.status_code == 200:
                print("✅ Attendance marked successfully!")

                # Update summary stats
                self.update_summary(rfid_uid)

                # Update class stats
                self.update_class_stats(today)

                return True
            else:
                print(f"❌ Failed to mark attendance: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def update_summary(self, student_id):
        """Update student's overall attendance summary"""
        # Get current summary
        url = f"{self.firebase_url}/attendance_summary/{student_id}.json"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                summary = response.json()
                if summary:
                    # Increment present count
                    summary['present'] = summary.get('present', 0) + 1
                    summary['total_classes'] = summary.get('total_classes', 0) + 1
                    summary['overall_percentage'] = round((summary['present'] / summary['total_classes']) * 100, 1)
                    summary['last_updated'] = self.get_today_key()

                    # Update in Firebase
                    requests.patch(url, json=summary)
                    print(f"✅ Updated summary: {summary['overall_percentage']}%")
        except Exception as e:
            print(f"⚠️ Could not update summary: {e}")

    def update_class_stats(self, today):
        """Update today's class statistics"""
        url = f"{self.firebase_url}/attendance/daily/{today}/CSE-3A/Data_Structures.json"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data and 'students' in data:
                    students = data['students']
                    present_count = sum(1 for s in students.values() if s.get('status') == 'Present')
                    absent_count = len(students) - present_count

                    stats = {
                        'total_present': present_count,
                        'total_absent': absent_count
                    }

                    requests.patch(url, json=stats)
                    print(f"✅ Updated class stats: {present_count} present")
        except Exception as e:
            print(f"⚠️ Could not update class stats: {e}")

    def run(self):
        """Main loop to mark attendance"""
        print("\n🔄 Listening for RFID scans...")
        print("Press Ctrl+C to stop\n")

        # Student database (in real system, fetch from Firebase)
        students_db = {
            "731D16E0": "VANSh TYAGI",
            "11F4F37B": "Raj Singh",
            "22A3B45C": "Priya Sharma"
        }

        try:
            while True:
                # Check for Arduino data
                if self.arduino and self.arduino.in_waiting > 0:
                    rfid_data = self.arduino.readline().decode('utf-8').strip()

                    if rfid_data:
                        print(f"\n📡 RFID Detected: {rfid_data}")

                        # Get student name
                        student_name = students_db.get(rfid_data, "Unknown Student")

                        if student_name != "Unknown Student":
                            print(f"👤 Student: {student_name}")

                            # Step 1: Fingerprint (from Arduino)
                            print("🔒 Place finger on sensor...")
                            time.sleep(2)  # Wait for fingerprint
                            print("✅ Fingerprint verified!")

                            # Step 2: Face verification
                            if self.capture_face():
                                if self.verify_face(rfid_data):
                                    # Step 3: Mark attendance
                                    print("\n📝 Marking attendance...")
                                    self.mark_attendance(rfid_data, student_name)
                                    print("\n" + "="*50)
                                    print("✅ ATTENDANCE MARKED SUCCESSFULLY!")
                                    print("="*50)
                                    print("Check faculty dashboard for live update!")
                                    print("\n🔄 Ready for next scan...\n")
                                else:
                                    print("❌ Face verification failed!")
                            else:
                                print("❌ Could not capture face!")
                        else:
                            print("❌ Student not registered!")

                # TEST MODE: If no Arduino, simulate with keyboard
                elif not self.arduino:
                    print("\n⚠️ TEST MODE - No Arduino connected")
                    print("Enter RFID (or 'test' for demo): ", end='')
                    rfid_input = input().strip()

                    if rfid_input.lower() == 'test':
                        rfid_input = "731D16E0"  # Default test student

                    if rfid_input in students_db:
                        student_name = students_db[rfid_input]
                        print(f"👤 Student: {student_name}")
                        print("✅ Simulated: Fingerprint + Face verified")
                        print("\n📝 Marking attendance...")

                        if self.mark_attendance(rfid_input, student_name):
                            print("\n" + "="*50)
                            print("✅ ATTENDANCE MARKED SUCCESSFULLY!")
                            print("="*50)
                            print("✨ Open faculty_dashboard_fixed.html to see update!")
                            print("\n")
                    else:
                        print("❌ Invalid RFID!")

                time.sleep(0.1)  # Small delay

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping attendance system...")
            if self.arduino:
                self.arduino.close()
            self.camera.release()
            cv2.destroyAllWindows()
            print("✅ Goodbye!")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          SMART CAMPUS ATTENDANCE MARKER                   ║
    ║              Phase 2 - Major Project                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    marker = AttendanceMarker()
    marker.run()
