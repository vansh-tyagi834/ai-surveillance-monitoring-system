import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import threading
import time
from datetime import datetime
from camera.firebase_upload import upload_snapshot

class AudioMonitor:
    def __init__(self):
        self.threshold = 0.15
        self.duration = 2
        self.samplerate = 44100
        self.suspicious_logs = []
        self.record_folder = "test_data/audio_snippets"
        self.last_detected_time = 0
        self.cooldown = 5
        os.makedirs(self.record_folder, exist_ok=True)

    def sound_alert(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = os.path.join(self.record_folder, f"suspicious_{timestamp}.wav")

        print("🎤 Recording suspicious sound...")
        recording = sd.rec(int(self.duration * self.samplerate), 
                           samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        wav.write(audio_path, self.samplerate, recording)

        # Capture snapshot if available
        snapshot_path = os.path.join("camera", "audio_alert.jpg")
        if os.path.exists(snapshot_path):
            upload_snapshot(snapshot_path)

        alert = f"🚨 Suspicious sound at {datetime.now().strftime('%H:%M:%S')}"
        self.suspicious_logs.append(alert)
        print(alert)

    def listen_audio(self):
        print("🔊 Audio monitoring started...")
        while True:
            try:
                audio = sd.rec(int(self.duration * self.samplerate), 
                               samplerate=self.samplerate, channels=1, dtype='float32')
                sd.wait()
                volume_norm = np.linalg.norm(audio)
                if volume_norm > self.threshold:
                    now = time.time()
                    if now - self.last_detected_time > self.cooldown:
                        self.sound_alert()
                        self.last_detected_time = now
            except Exception as e:
                print(f"❌ Audio error: {e}")

    def start(self):
        threading.Thread(target=self.listen_audio, daemon=True).start()

    def get_logs(self):
        return self.suspicious_logs
