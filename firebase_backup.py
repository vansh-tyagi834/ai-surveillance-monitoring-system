# Run this to check Firebase data
import requests

firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"
response = requests.get(f"{firebase_url}/exam_alerts.json")
data = response.json()

print(f"Total alerts in Firebase: {len(data) if data else 0}")
if data:
    latest = list(data.items())[-3:]
    for key, alert in latest:
        print(f"Latest: {alert['timestamp']} - {alert['type']}")
