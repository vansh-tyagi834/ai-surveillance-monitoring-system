import requests
import json
import time

def test_firebase_direct():
    """Direct Firebase test without SDK"""
    
    firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com/rfid_scans.json"
    
    print("🔥 Testing Firebase connection...")
    print(f"🔗 URL: {firebase_url}")
    
    try:
        # GET request to read data
        response = requests.get(firebase_url)
        
        print(f"📊 Response Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Firebase connection SUCCESS!")
            
            if data:
                print(f"📄 Found {len(data)} RFID scans:")
                
                # Show latest 3 scans
                count = 0
                for key, scan in data.items():
                    if count < 3:
                        print(f"  🏷️  Key: {key}")
                        print(f"  📱 Student ID: {scan.get('student_id')}")
                        print(f"  ⏰ Timestamp: {scan.get('timestamp')}")
                        print(f"  📍 Status: {scan.get('status')}")
                        print("  " + "-"*30)
                        count += 1
                    
                return True
            else:
                print("📭 No data found in Firebase")
                return False
                
        else:
            print(f"❌ Firebase connection FAILED: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def monitor_rfid_scans():
    """Monitor new RFID scans"""
    
    firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com/rfid_scans.json"
    
    print("\n" + "="*50)
    print("🔍 MONITORING RFID SCANS...")
    print("="*50)
    print("📋 Instructions:")
    print("1. Keep this script running")
    print("2. Scan RFID card on Arduino")
    print("3. Watch for new detections")
    print("4. Press Ctrl+C to stop")
    print("="*50)
    
    last_scan_count = 0
    
    while True:
        try:
            response = requests.get(firebase_url)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    current_scan_count = len(data)
                    
                    # Check if new scan added
                    if current_scan_count > last_scan_count:
                        print(f"\n🚨 NEW RFID SCAN DETECTED!")
                        
                        # Get latest scan
                        latest_key = list(data.keys())[-1]
                        latest_scan = data[latest_key]
                        
                        print(f"📱 Student ID: {latest_scan.get('student_id')}")
                        print(f"⏰ Timestamp: {latest_scan.get('timestamp')}")
                        print(f"🔑 Firebase Key: {latest_key}")
                        print(f"📊 Total Scans: {current_scan_count}")
                        
                        # Calculate time difference
                        arduino_time = latest_scan.get('timestamp', 0)
                        current_time = time.time() * 1000  # Convert to milliseconds
                        time_diff = (current_time - arduino_time) / 1000  # Convert to seconds
                        
                        print(f"🕐 Time difference: {time_diff:.1f} seconds")
                        
                        print("✅ Scan detection SUCCESS!")
                        print("-" * 50)
                        
                        last_scan_count = current_scan_count
                    else:
                        print(f"🔄 Checking... ({current_scan_count} total scans)")
                else:
                    print("📭 No data found")
            else:
                print(f"❌ Request failed: {response.status_code}")
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n❌ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    print("🎯 FIREBASE SYNC TEST")
    print("="*50)
    
    # Step 1: Test connection
    if test_firebase_direct():
        print("\n✅ Connection test passed!")
        
        # Step 2: Monitor scans
        input("\nPress Enter to start monitoring RFID scans...")
        monitor_rfid_scans()
    else:
        print("\n❌ Connection test failed!")
        print("Check your Firebase URL and internet connection")