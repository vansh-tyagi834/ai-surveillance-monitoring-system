
"""
🎯 STEP 1: FIREBASE MONITORING FOR ARDUINO ATTENDANCE
=====================================================

Add this to your existing NASSCOM Python code to monitor
Arduino attendance records from Firebase in real-time.

This is a MINIMAL addition - we'll add more in later steps.
"""

import requests
import time
import threading
from datetime import datetime

class ArduinoAttendanceMonitor:
    """Monitor Firebase for Arduino attendance records"""

    def __init__(self):
        # Firebase URL (same as your existing code)
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"

        # Track last checked record
        self.last_record_id = None

        # Monitoring flag
        self.monitoring = False

        print("✅ Arduino Attendance Monitor initialized")

    def start_monitoring(self):
        """Start monitoring Firebase for new attendance records"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Started monitoring Arduino attendance records...")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("⏸️ Stopped monitoring")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Check for new records
                new_record = self._check_new_attendance()

                if new_record:
                    # New attendance detected!
                    self._handle_new_attendance(new_record)

                # Check every 2 seconds
                time.sleep(2)

            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(5)

    def _check_new_attendance(self):
        """Check Firebase for new attendance records"""
        try:
            # Get all attendance records
            url = f"{self.firebase_url}/attendance_records.json"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if data:
                    # Get most recent record
                    latest_id = list(data.keys())[-1]

                    # Check if this is a new record
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
        """Handle new attendance record from Arduino"""
        record_id = record['id']
        data = record['data']

        # Extract data
        student_id = data.get('student_id', 'Unknown')
        fingerprint_id = data.get('fingerprint_id', 0)
        confidence = data.get('confidence', 0)
        status = data.get('status', 'unknown')
        device = data.get('device', 'Unknown')

        # Print notification
        print("\n" + "="*60)
        print("🎯 NEW ARDUINO ATTENDANCE DETECTED!")
        print("="*60)
        print(f"📱 Card ID: {student_id}")
        print(f"👆 Fingerprint ID: {fingerprint_id}")
        print(f"📊 Confidence: {confidence}")
        print(f"✅ Status: {status}")
        print(f"🔧 Device: {device}")
        print(f"🆔 Record ID: {record_id}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # TODO: In next steps, we'll add:
        # - Camera capture
        # - Face recognition
        # - Complete attendance logging

        return record


# ============================================
# STEP 1 INTEGRATION INSTRUCTIONS
# ============================================

"""
HOW TO ADD THIS TO YOUR EXISTING CODE:
======================================

1. Open your existing NASSCOM Python file (gpttt.txt or similar)

2. Find the __init__ method of IntegratedProctoringSystem class

3. Add this line after existing initialization:

   self.arduino_monitor = ArduinoAttendanceMonitor()

4. Find the start() or run() method

5. Add this line to start monitoring:

   self.arduino_monitor.start_monitoring()

6. Run your Python code

7. Scan RFID + Fingerprint on Arduino

8. Python should print "NEW ARDUINO ATTENDANCE DETECTED!"

That's it for STEP 1! Test this before we move to STEP 2.
"""


# ============================================
# TESTING CODE (RUN THIS SEPARATELY FIRST)
# ============================================

if __name__ == "__main__":
    print("\n🎯 TESTING ARDUINO ATTENDANCE MONITOR")
    print("="*60)

    # Create monitor
    monitor = ArduinoAttendanceMonitor()

    # Start monitoring
    monitor.start_monitoring()

    print("\n📋 Instructions:")
    print("1. Make sure Arduino is running and connected")
    print("2. Scan RFID card on Arduino")
    print("3. Place finger on fingerprint sensor")
    print("4. Watch for notification here!")
    print("\n⏳ Monitoring... (Press Ctrl+C to stop)")

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏸️ Stopping monitor...")
        monitor.stop_monitoring()
        print("✅ Monitor stopped!")
