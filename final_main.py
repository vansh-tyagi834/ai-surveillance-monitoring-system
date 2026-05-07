import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import requests
import mediapipe as mp
import numpy as np
import threading


class IntegratedProctoringSystem:
    def __init__(self):
        # 🎯 Camera setup
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 🔥 Firebase configuration
        self.firebase_url = "https://ai-proctored-exam-system-default-rtdb.firebaseio.com"
        
        # 🧠 Face recognition setup
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}
        self.student_names = {}
        self.student_database = {}  # Store complete student info
        
        # 📊 Head pose setup
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 🚨 Monitoring variables
        self.current_student = None
        self.current_student_info = None
        self.exam_session_active = False
        self.session_start_time = 0
        self.suspicious_activities = []
        
        # 📈 Detection counters
        self.no_face_count = 0
        self.unknown_face_count = 0
        self.head_down_count = 0
        self.looking_side_count = 0
        self.multiple_face_count = 0
        
        # 🎯 Thresholds
        self.CONFIDENCE_THRESHOLD = 165  # Increased for better recognition
        self.NO_FACE_THRESHOLD = 60
        self.UNKNOWN_FACE_THRESHOLD = 30
        self.HEAD_DOWN_THRESHOLD = 90
        self.LOOKING_SIDE_THRESHOLD = 60
        self.MULTIPLE_FACE_THRESHOLD = 45
        
        # 📱 RFID monitoring
        self.last_scan_count = 0
        self.rfid_check_interval = 2  # seconds
        self.last_rfid_check = 0
        
        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 8000
        
        # Alert cooldown
        self.last_alert_time = {}
        self.alert_cooldown = {
            'UNUSUAL_HEAD_TILT': 15,     # 15 seconds between head tilt alerts
            'WRONG_STUDENT': 20,         # 20 seconds between identity alerts  
            'UNKNOWN_PERSON': 10,        # 10 seconds between unknown person alerts
            'STUDENT_ABSENT_WARNING': 8 
        }
        
        # Registration mode
        self.registration_mode = False
        self.registration_photos = []
        self.registration_card_id = None
        
        # Initialize system
        print("🚀 INITIALIZING INTEGRATED PROCTORING SYSTEM...")
        self.init_camera()
        self.load_student_database()
        self.create_directories()
        # Add to __init__()
        self.head_positions = []  # Track head center over time
        self.MOVEMENT_THRESHOLD_FRAMES = 40  # ~1.3 seconds at 30fps
       
        # 🧠 SMART THRESHOLD SYSTEM
        self.smart_thresholds = {
            'HEAD_TILT': {
                'duration_threshold': 5.0,  # 3 seconds sustained behavior
                'intensity_threshold': 35.0,  # degrees
                'warning_threshold': 4.0,   # yellow alert at 2s
                'alert_threshold': 6.0,     # red alert at 4s
                'current_duration': 0.0,
                'last_detection_time': 0,
                'consecutive_detections': 0
            },
            'LOOKING_AWAY': {
                'duration_threshold': 4.0,
                'intensity_threshold': 50.0,  # degrees yaw
                'warning_threshold': 3.0,
                'alert_threshold': 5.0,
                'current_duration': 0.0,
                'last_detection_time': 0,
                'consecutive_detections': 0
            },
            'HEAD_DOWN': {
                'duration_threshold': 4.0,
                'intensity_threshold': -30.0,  # negative pitch
                'warning_threshold': 3.0,
                'alert_threshold': 5.0,
                'current_duration': 0.0,
                'last_detection_time': 0,
                'consecutive_detections': 0
            },
            'RESTLESS_BEHAVIOR': {
                'duration_threshold': 4.0,
                'movement_threshold': 80.0,  # pixels movement
                'warning_threshold': 2.5,
                'alert_threshold': 5.0,
                'current_duration': 0.0,
                'last_detection_time': 0,
                'consecutive_detections': 0
            },
            'NO_FACE': {
                'duration_threshold': 3.0,
                'warning_threshold': 2.0,
                'alert_threshold': 4.0,
                'current_duration': 0.0,
                'last_detection_time': 0,
                'consecutive_detections': 0
            }
        }

        # 🎯 ALERT LEVEL SYSTEM
        self.alert_levels = {
            'NORMAL': {'color': (0, 255, 0), 'text': 'NORMAL'},
            'YELLOW': {'color': (0, 255, 255), 'text': 'WARNING'},
            'ORANGE': {'color': (0, 165, 255), 'text': 'CAUTION'},
            'RED': {'color': (0, 0, 255), 'text': 'ALERT'}
        }

        # 🧠 BEHAVIORAL INTELLIGENCE
        self.behavioral_context = {
            'exam_phase': 'reading',  # reading, writing, thinking
            'activity_intensity': 0.0,  # 0-100 scale
            'baseline_movement': 0.0,
            'stress_indicators': [],
            'phase_start_time': 0
        }
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only track one hand for gestures
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Toilet break gesture tracking
        self.toilet_break_gesture = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False,
            'break_active': False,
            'break_start_time': 0,
            'break_duration': 90,  # 1.5 minutes default
            'detection_threshold': 3.0,  # 3 seconds to confirm gesture
        }
        self.question_gesture = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False,
            'detection_threshold': 2.0,  # 2 seconds to confirm
            'cooldown_period': 30.0  # 30 seconds between questions
        }

        self.technical_issue_gesture = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False,
            'detection_threshold': 3.0,  # 3 seconds to confirm
            'issue_reported': False,
            'report_time': 0
        }

        self.ready_gesture = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False,
            'detection_threshold': 2.0,  # 2 seconds to confirm
            'last_ready_time': 0
        }

        # Gesture tracking system
        self.last_gesture_time = {}
        self.active_gestures = []


        # 📊 MOVEMENT INTENSITY TRACKING
        self.movement_history = []
        self.movement_baseline_frames = 300  # 10 seconds at 30fps
        self.fps = 30  # approximate
        self.arduino_monitor_active = False
        print("✅ Arduino monitor ready")
        self.face_verification_in_progress = False
        self.current_rfid_scan = None


    def detect_toilet_break_gesture(self, frame, hands_results):
        
        """Detect raised hand gesture for toilet break request"""
        current_time = time.time()

        if self.toilet_break_gesture['break_active']:
            return False
    
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Get landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
            
                # Check if hand is raised (wrist below middle finger tip)
                if len(landmarks) >= 21:  # MediaPipe gives 21 landmarks
                    wrist_y = landmarks[0][1]  # Wrist position
                    middle_tip_y = landmarks[12][1]  # Middle finger tip
                
                    # Hand is raised if fingers are above wrist
                    hand_raised = middle_tip_y < wrist_y - 50
                
                    if hand_raised:
                        if not self.toilet_break_gesture['active']:
                            self.toilet_break_gesture['active'] = True
                            self.toilet_break_gesture['start_time'] = current_time
                            print("🤚 Hand raised - monitoring for toilet break gesture...")
                    
                        # Update duration
                        self.toilet_break_gesture['duration'] = current_time - self.toilet_break_gesture['start_time']
                    
                        # Check if gesture held long enough
                        if (self.toilet_break_gesture['duration'] >= 3.0 and
                            not self.toilet_break_gesture['confirmed']):
                        
                            self.toilet_break_gesture['confirmed'] = True
                            self.start_toilet_break()
                            return True
                        
                    else:
                        self.reset_toilet_break_gesture()
        else:
            self.reset_toilet_break_gesture()
    
        return False
    

    def reset_toilet_break_gesture(self):
        """Reset toilet break gesture tracking"""
        if self.toilet_break_gesture['active'] and not self.toilet_break_gesture['confirmed']:
            print("🤚 Hand gesture reset - gesture not completed")
    
        self.toilet_break_gesture.update({
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False
        })

    def start_toilet_break(self):
        """Start toilet break timer"""
        print("🚽 TOILET BREAK REQUEST CONFIRMED!")
        print(f"✅ Break approved for {self.toilet_break_gesture['break_duration']} seconds")
    
        self.toilet_break_gesture['break_active'] = True
        self.toilet_break_gesture['break_start_time'] = time.time()
    
        # Log the break request
        self.log_suspicious_activity("TOILET_BREAK_REQUEST", 
                                    f"Student requested {self.toilet_break_gesture['break_duration']}s toilet break")
    
        # Reset gesture detection
        self.reset_toilet_break_gesture()

    def check_toilet_break_status(self):
        """Check if toilet break is active and handle return"""
        if not self.toilet_break_gesture['break_active']:
            return False
    
        current_time = time.time()
        elapsed = current_time - self.toilet_break_gesture['break_start_time']
        remaining = self.toilet_break_gesture['break_duration'] - elapsed
    
        if remaining <= 0:
            # Break time expired
            print("⏰ TOILET BREAK TIME EXPIRED!")
            self.log_suspicious_activity("TOILET_BREAK_EXPIRED", 
                                    f"Break time exceeded by {abs(remaining):.1f}s")
            self.toilet_break_gesture['break_active'] = False
            return False
    
        return True
    
    def detect_advanced_gestures(self, frame, hands_results):
        """Detect advanced gestures: Question, Technical Issue, Ready"""
        current_time = time.time()
        gesture_detected = False
    
        # Skip if toilet break is active
        if self.toilet_break_gesture['break_active']:
            return False
    
        if hands_results.multi_hand_landmarks:
            hand_count = len(hands_results.multi_hand_landmarks)
        
            if hand_count == 1:
                # Single hand gestures
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                landmarks = self.extract_hand_landmarks(hand_landmarks, frame)
            
                # Check for Question Gesture (Index finger raised)
                if self.is_index_finger_raised(landmarks):
                    gesture_detected = self.process_question_gesture(current_time)
            
                # Check for Ready Gesture (Thumbs up)  
                elif self.is_thumbs_up(landmarks):
                    gesture_detected = self.process_ready_gesture(current_time)
                else:
                    self.reset_single_hand_gestures()
                
            elif hand_count == 2:
                # Both hands raised for technical issue
                if self.are_both_hands_raised(hands_results.multi_hand_landmarks, frame):
                    gesture_detected = self.process_technical_issue_gesture(current_time)
                else:
                    self.reset_technical_issue_gesture()
        else:
            self.reset_all_advanced_gestures()
    
        return gesture_detected
    
    def is_index_finger_raised(self, landmarks):
        """Check if index finger is raised (Question gesture)"""
        if len(landmarks) < 21:
            return False
    
        wrist = landmarks[0]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
    
        # Index finger extended, others folded, finger above wrist
        index_extended = index_tip[1] < index_pip[1] - 20
        other_fingers_folded = (middle_tip[1] > index_tip[1] + 30 and
                               ring_tip[1] > index_tip[1] + 30 and
                               pinky_tip[1] > index_tip[1] + 30)
        finger_raised = index_tip[1] < wrist[1] - 80
    
        return index_extended and other_fingers_folded and finger_raised

    def is_thumbs_up(self, landmarks):
        """Check for thumbs up gesture (Ready/OK)"""
        if len(landmarks) < 21:
            return False
    
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        wrist = landmarks[0]
    
        # Thumb up, other fingers folded
        thumb_up = thumb_tip[1] < thumb_ip[1] - 20 and thumb_tip[1] < wrist[1] - 50
        fingers_folded = index_tip[1] > wrist[1] - 20
    
        return thumb_up and fingers_folded

    def extract_hand_landmarks(self, hand_landmarks, frame):
        """Extract hand landmark coordinates"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            landmarks.append((x, y))
        return landmarks

    def are_both_hands_raised(self, multi_hand_landmarks, frame):
        """Check if both hands are raised (Technical Issue gesture)"""
        if len(multi_hand_landmarks) != 2:
            return False
    
        both_hands_raised = True
        for hand_landmarks in multi_hand_landmarks:
            landmarks = self.extract_hand_landmarks(hand_landmarks, frame)
            wrist_y = landmarks[0][1]
            middle_tip_y = landmarks[12][1]
        
            if not (middle_tip_y < wrist_y - 60):
                both_hands_raised = False
                break
    
        return both_hands_raised


    def process_question_gesture(self, current_time):
        if not self.question_gesture['active']:
            self.question_gesture['active'] = True
            self.question_gesture['start_time'] = current_time
            print("❓ Index finger raised - monitoring for question gesture...")
    
        self.question_gesture['duration'] = current_time - self.question_gesture['start_time']
    
        if (self.question_gesture['duration'] >= self.question_gesture['detection_threshold'] and
            not self.question_gesture['confirmed']):
        
            last_question = self.last_gesture_time.get('question', 0)
            if current_time - last_question >= self.question_gesture['cooldown_period']:
                self.question_gesture['confirmed'] = True
                self.confirm_question_gesture(current_time)
                return True
    
        return False

    def process_ready_gesture(self, current_time):
        if not self.ready_gesture['active']:
            self.ready_gesture['active'] = True
            self.ready_gesture['start_time'] = current_time
            print("✅ Thumbs up detected - monitoring for ready gesture...")
    
        self.ready_gesture['duration'] = current_time - self.ready_gesture['start_time']
    
        if (self.ready_gesture['duration'] >= self.ready_gesture['detection_threshold'] and
            not self.ready_gesture['confirmed']):
        
            self.ready_gesture['confirmed'] = True
            self.confirm_ready_gesture(current_time)
            return True
    
        return False

    def process_technical_issue_gesture(self, current_time):
        if not self.technical_issue_gesture['active']:
            self.technical_issue_gesture['active'] = True
            self.technical_issue_gesture['start_time'] = current_time
            print("📝 Both hands raised - monitoring for technical issue gesture...")
    
        self.technical_issue_gesture['duration'] = current_time - self.technical_issue_gesture['start_time']
    
        if (self.technical_issue_gesture['duration'] >= self.technical_issue_gesture['detection_threshold'] and
            not self.technical_issue_gesture['confirmed']):
        
            self.technical_issue_gesture['confirmed'] = True
            self.confirm_technical_issue_gesture(current_time)
            return True
    
        return False
    
    def confirm_question_gesture(self, current_time):
        print("❓ QUESTION REQUEST CONFIRMED!")
        print("✅ Teacher notification sent - Student has a question")
    
        self.log_suspicious_activity("QUESTION_REQUEST", 
                                "Student raised index finger - requesting teacher assistance")
    
        self.last_gesture_time['question'] = current_time
        self.reset_question_gesture()

    def confirm_ready_gesture(self, current_time):
        print("✅ READY GESTURE CONFIRMED!")
        print("🎯 Student indicates readiness/understanding")
    
        self.log_suspicious_activity("STUDENT_READY", 
                                    "Student gave thumbs up - indicates readiness")
    
        self.ready_gesture['last_ready_time'] = current_time
        self.reset_ready_gesture()

    def confirm_technical_issue_gesture(self, current_time):
        print("📝 TECHNICAL ISSUE REPORTED!")
        print("🚨 System administrator notified - Technical assistance required")
    
        self.log_suspicious_activity("TECHNICAL_ISSUE", 
                                    "Student raised both hands - reporting technical problem")
    
        self.technical_issue_gesture['issue_reported'] = True
        self.technical_issue_gesture['report_time'] = current_time
        self.reset_technical_issue_gesture()


    def reset_question_gesture(self):
        if self.question_gesture['active'] and not self.question_gesture['confirmed']:
            print("❓ Question gesture reset")
    
        self.question_gesture.update({
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False
        })

    def reset_ready_gesture(self):
        if self.ready_gesture['active'] and not self.ready_gesture['confirmed']:
            print("✅ Ready gesture reset")
    
        self.ready_gesture.update({
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False
        })

    def reset_technical_issue_gesture(self):
        if self.technical_issue_gesture['active'] and not self.technical_issue_gesture['confirmed']:
            print("📝 Technical issue gesture reset")
    
        self.technical_issue_gesture.update({
            'active': False,
            'start_time': 0,
            'duration': 0,
            'confirmed': False
        })

    def reset_single_hand_gestures(self):
        self.reset_question_gesture()
        self.reset_ready_gesture()

    def reset_all_advanced_gestures(self):
        self.reset_question_gesture()
        self.reset_ready_gesture()
        self.reset_technical_issue_gesture()






    def draw_gesture_overlay(self, frame, hands_results):
        """Draw gesture detection overlay on frame"""
        # Draw hand landmarks if detected
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
        # Show gesture status
        if self.toilet_break_gesture['active']:
            progress = self.toilet_break_gesture['duration']
            threshold = self.toilet_break_gesture['detection_threshold']
        
            # Progress bar for gesture
            bar_width = int(300 * min(progress / threshold, 1.0))
            cv2.rectangle(frame, (10, 50), (310, 70), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), (0, 255, 255), -1)
        
            # Status text
            cv2.putText(frame, f"Toilet Break Gesture: {progress:.1f}s / {threshold}s", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
            if progress >= threshold:
                cv2.putText(frame, "TOILET BREAK APPROVED!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Show break status
        if self.toilet_break_gesture['break_active']:
            current_time = time.time()
            elapsed = current_time - self.toilet_break_gesture['break_start_time']
            remaining = self.toilet_break_gesture['break_duration'] - elapsed
        
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 100, 0), -1)
            cv2.putText(frame, "TOILET BREAK ACTIVE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Time Remaining: {remaining:.0f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)



    def detect_behavioral_cheating(self, faces, frame):
        """Detect looking around, restless behavior, etc."""
        current_time = time.time()

        if len(faces) == 1:
            x, y, w, h = faces[0]
            head_center = (x + w//2, y + h//2)

        # Store position
            self.head_positions.append({
                'position': head_center,
                'timestamp': current_time
            })

        # Keep last 3 seconds
            self.head_positions = [
                pos for pos in self.head_positions
                if current_time - pos['timestamp'] < 3.0
            ]

        # Analyze if enough data
            if len(self.head_positions) >= 10:
                positions = [p['position'] for p in self.head_positions]
                x_vals = [p[0] for p in positions]
                y_vals = [p[1] for p in positions]

                x_range = max(x_vals) - min(x_vals)
                y_range = max(y_vals) - min(y_vals)

            # 1. Looking around (left-right)
                if x_range > 100:
                    cv2.putText(frame, "⚠️ LOOKING AROUND", (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    #self.log_suspicious_activity("LOOKING_AROUND", f"Head moved {x_range:.1f}px left/right")

            # 2. Up-down movement
                if y_range > 80:
                    cv2.putText(frame, "⚠️ HEAD MOVEMENT UP/DOWN", (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    #self.log_suspicious_activity("HEAD_UP_DOWN", f"Head moved {y_range:.1f}px vertically")

            # 3. Restless/jittery
                if len(positions) > 8 and (x_range > 60 or y_range > 50):
                    cv2.putText(frame, "⚠️ RESTLESS BEHAVIOR", (10, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 165), 2)
                    #self.log_suspicious_activity("RESTLESS_BEHAVIOR", "Jittery head movement")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['student_faces', 'student_data', 'exam_reports', 'registration_photos']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
    
    def init_camera(self):
        """Initialize camera – prefer DroidCam on index 1"""

        # Reset cap
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None

        # ⚙️ Hum pehle DroidCam (1), fir 0, fir 2 check karenge
        preferred_indexes = [1, 0, 2]

        for idx in preferred_indexes:
            cap = cv2.VideoCapture(idx)   # ❌ NO CAP_DSHOW, NO MSMF, NOTHING
            if not cap.isOpened():
                cap.release()
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue

            # ✅ Success: camera mil gaya
            self.cap = cap
            height, width = frame.shape[:2]

            # Head pose ke liye camera matrix
            self.camera_matrix = np.array([
                [width * 0.8, 0.0, width / 2.0],
                [0.0, width * 0.8, height / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            print(f"✅ Camera ready on index {idx}: {width}x{height}")
            return True

        print("❌ No camera found on indexes [1, 0, 2] (DroidCam not connected?)")
        return False
    
    def load_student_database(self):
        """Load complete student database with enhanced info"""
        if not os.path.exists('student_data'):
            print("📁 Creating student data directory...")
            return False
        
        faces = []
        labels = []
        student_id_map = {}
        label_counter = 0
        
        print("📚 Loading student database...")
        
        for filename in os.listdir('student_data'):
            if filename.endswith('_registration.json'):
                filepath = os.path.join('student_data', filename)
                
                try:
                    with open(filepath, 'r') as f:
                        student_data = json.load(f)
                    
                    student_id = student_data.get('student_id')
                    student_name = student_data.get('name', 'Unknown')
                    roll_number = student_data.get('roll_number', 'Unknown')
                    photo_paths = student_data.get('photo_paths', [])
                    
                    if student_id and photo_paths:
                        student_faces_loaded = 0
                        
                        # Store complete student info
                        self.student_database[student_id] = {
                            'name': student_name,
                            'roll_number': roll_number,
                            'registration_date': student_data.get('registration_date'),
                            'photo_paths': photo_paths
                        }
                        
                        for photo_path in photo_paths:
                            if os.path.exists(photo_path):
                                img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img_resized = cv2.resize(img, (100, 100))
                                    faces.append(img_resized)
                                    labels.append(label_counter)
                                    student_faces_loaded += 1
                        
                        if student_faces_loaded > 0:
                            student_id_map[label_counter] = student_id
                            self.student_names[label_counter] = f"{student_name} ({student_id})"
                            label_counter += 1
                            print(f"✅ Loaded {student_faces_loaded} faces for {student_name} - {student_id}")
                
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        if len(faces) > 0:
            print(f"🧠 Training face recognizer with {len(faces)} samples...")
            self.face_recognizer.train(faces, np.array(labels))
            self.known_faces = student_id_map
            print(f"✅ Face recognition ready! Students: {len(student_id_map)}")
            return True
        else:
            print("📝 No registered students found!")
            return False
    
    def check_rfid_scans(self):
        """Check for new RFID scans from Firebase"""
        current_time = time.time()
        
        # Throttle RFID checks
        if current_time - self.last_rfid_check < self.rfid_check_interval:
            return None
        
        self.last_rfid_check = current_time
        
        try:
            response = requests.get(f"{self.firebase_url}/rfid_scans.json", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > self.last_scan_count:
                    # Get latest scan
                    latest_key = list(data.keys())[-1]
                    latest_scan = data[latest_key]
                    
                    self.last_scan_count = len(data)
                    return latest_scan
            
            return None
            
        except Exception as e:
            print(f"⚠️ Firebase connection error: {e}")
            return None
    
    def register_new_student(self, card_id):
        """Complete student registration process"""
        print(f"\n🆕 NEW STUDENT REGISTRATION")
        print("="*50)
        print(f"📱 Card ID: {card_id}")
        
        # Get student details
        name = input("👤 Enter student name: ").strip()
        roll_number = input("🔢 Enter roll number: ").strip()
        
        if not name or not roll_number:
            print("❌ Name and roll number are required!")
            return False
        
        print(f"\n📸 PHOTO CAPTURE PROCESS")
        print("="*40)
        print("📋 Instructions:")
        print("   1. Look straight at camera")
        print("   2. Press SPACE to capture photo")
        print("   3. Capture 3 photos for better recognition")
        print("   4. Press ESC to cancel registration")
        
        self.registration_mode = True
        self.registration_photos = []
        self.registration_card_id = card_id
        
        photo_count = 0
        target_photos = 3
        
        while photo_count < target_photos and self.registration_mode:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Display registration info
            cv2.putText(frame, f"REGISTRATION MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Student: {name}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll_number}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Card: {card_id}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Photos: {photo_count}/{target_photos}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Detect face for registration
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(120, 120))
            
            # Check if this face already exists
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Check against existing faces
                face_roi = gray[y:y+h, x:x+w]
                existing_student, confidence = self.recognize_face(face_roi)
                
                if existing_student and confidence < 80:
                    # Face already registered!
                    existing_info = self.student_database.get(existing_student, {})
                    cv2.putText(frame, f"⚠️ ALREADY REGISTERED!", (x, y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Name: {existing_info.get('name', 'Unknown')}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Card: {existing_student}", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Roll: {existing_info.get('roll_number', 'Unknown')}", (x, y+h+40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "✅ Ready to capture", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            elif len(faces) > 1:
                cv2.putText(frame, "⚠️ Multiple faces! Position properly", (10, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            elif len(faces) == 0:
                cv2.putText(frame, "⚠️ No face detected!", (10, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("📸 Student Registration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(faces) == 1:  # SPACE key
                x, y, w, h = faces[0]
                # Check for duplicate again before capturing
                face_roi = gray[y:y+h, x:x+w]
                existing_student, confidence = self.recognize_face(face_roi)
                
                if existing_student and confidence < 80:
                    existing_info = self.student_database.get(existing_student, {})
                    print(f"\n❌ DUPLICATE DETECTED!")
                    print(f"This person is already registered as:")
                    print(f"   Name: {existing_info.get('name', 'Unknown')}")
                    print(f"   Roll: {existing_info.get('roll_number', 'Unknown')}")
                    print(f"   Card: {existing_student}")
                    
                    choice = input("\nOptions:\n1. Continue with new registration (y)\n2. Cancel registration (n)\nChoice: ").lower()
                    
                    if choice != 'y':
                        print("❌ Registration cancelled due to duplicate!")
                        self.registration_mode = False
                        cv2.destroyWindow("📸 Student Registration")
                        return False
                
                # ✅ CROP + RESIZE FACE ONLY
                face_gray = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_gray, (100, 100))

                photo_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_filename = f"registration_photos/{card_id}_{name}_{photo_count}_{timestamp}.jpg"

                cv2.imwrite(photo_filename, face_resized)
                self.registration_photos.append(photo_filename)

                print(f"📸 Photo {photo_count}/{target_photos} captured: {photo_filename}")
                time.sleep(1)

                # 🔥🔥 CHANGE ENDS HERE 🔥🔥

            elif key == 27:  # ESC key
                print("❌ Registration cancelled by user!")
                self.registration_mode = False
                break
    
        cv2.destroyWindow("📸 Student Registration")
    
        if photo_count == target_photos:
            return self.complete_registration(card_id, name, roll_number)
        else:
            print("❌ Registration incomplete!")
            return False

   


    
    def complete_registration(self, card_id, name, roll_number):
        """Complete the student registration process"""
        try:
            # Create student data
            student_data = {
                'student_id': card_id,
                'name': name,
                'roll_number': roll_number,
                'registration_date': datetime.now().isoformat(),
                'photo_paths': self.registration_photos,
                'registration_complete': True
            }
            
            # Save to local JSON
            json_filename = f"student_data/{card_id}_registration.json"
            with open(json_filename, 'w') as f:
                json.dump(student_data, f, indent=2)
            
            # Update student database
            self.student_database[card_id] = {
                'name': name,
                'roll_number': roll_number,
                'registration_date': student_data['registration_date'],
                'photo_paths': self.registration_photos
            }
            
            # Send to Firebase
            try:
                firebase_url = f"{self.firebase_url}/students/{card_id}.json"
                requests.put(firebase_url, json=student_data, timeout=10)
                print("🔥 Student data uploaded to Firebase!")
            except Exception as e:
                print(f"⚠️ Firebase upload failed: {e}")
            
            print(f"\n✅ REGISTRATION COMPLETE!")
            print(f"   👤 Name: {name}")
            print(f"   🔢 Roll: {roll_number}")
            print(f"   📱 Card: {card_id}")
            print(f"   📸 Photos: {len(self.registration_photos)}")
            
            # Reload face recognition with new student
            self.load_student_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False
    
    def handle_rfid_scan(self, scan_data):
        """Handle RFID scan with proper registration flow"""
        card_id = scan_data.get('student_id')
        scan_time = scan_data.get('timestamp', datetime.now().isoformat())
        print(f"\n🚨 RFID SCAN DETECTED: {card_id}")
        print(f"⏰ Scan time: {scan_time}")

    # Check if student exists in database
        if card_id in self.student_database:
        # Existing student
            student_info = self.student_database[card_id]
            reg_date = student_info.get('registration_date', 'Unknown')
            reg_display = reg_date[:10] if reg_date and reg_date != 'Unknown' else 'Unknown'

            print(f"\n✅ REGISTERED STUDENT FOUND:")
            print(f"   👤 Name: {student_info.get('name', 'Unknown')}")
            print(f"   🔢 Roll: {student_info.get('roll_number', 'Unknown')}")
            print(f"   📱 Card: {card_id}")
            print(f"   📅 Registered: {reg_display}")

            print(f"\n🎯 OPTIONS:")
            print(f"1. Start exam session (s)")
            print(f"2. Re-register student (r)")
            print(f"3. Cancel (c)")
            choice = input("Enter choice: ").lower()

            if choice == 's':
                self.current_student = card_id
                self.current_student_info = student_info  # Includes name, roll, etc.
                self.exam_session_active = True
                self.session_start_time = time.time()
                print(f"🚀 EXAM SESSION STARTED for {student_info.get('name', 'Unknown')}")
                return True
            elif choice == 'r':
                print(f"🔄 Re-registering {student_info.get('name', 'Unknown')}...")
                return self.register_new_student(card_id)
            else:
                print("❌ Operation cancelled")
                return False
        else:
            # New student - needs registration
            print(f"\n🆕 NEW CARD DETECTED: {card_id}")
            print("📝 This card is not registered in the system.")
            register_choice = input("Register new student? (y/n): ").lower()
            if register_choice == 'y':
                return self.register_new_student(card_id)
            else:
                print("❌ Registration cancelled")
                return False
    
    def wait_for_rfid_scan(self):
        """Wait for RFID scan to start session or registration"""
        print("\n🔍 RFID SCAN MONITOR")
        print("="*60)
        print("📡 Waiting for card scan on Arduino...")
        print("🎯 Scan your RFID card to:")
        print("   • Start exam session (if registered)")
        print("   • Begin new student registration")
        print("📋 Press 'q' to quit system")
        print("="*60)

        last_scan_id = None
        wait_count = 0

        
        # Reset scan counter to detect new scans
        try:
            response = requests.get(f"{self.firebase_url}/rfid_scans.json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.last_scan_count = len(data) if data else 0
        except:
            self.last_scan_count = 0
        
        while not self.exam_session_active:

    


            wait_count += 1
            # Check for RFID scan
            scan_data = self.check_rfid_scans()
            
            if scan_data:
                return self.handle_rfid_scan(scan_data)
            
            # Check for manual quit
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                print("❌ System shutdown requested")
                return False
            
            # Show waiting message periodically
        #    if int(time.time()) % 10 == 0:
        #        print("⏳ Still waiting for RFID scan...")
        
        return True
    
    def recognize_face(self, face_roi):
        """Recognize face using trained model"""
        if not self.known_faces:
            return None, 0
        
        try:
            face_resized = cv2.resize(face_roi, (100, 100))
            # print(f"DEBUG: recognize_face called, face_resized shape: {face_resized.shape}") # ADD THIS LINE
            label, confidence = self.face_recognizer.predict(face_resized)
            
            # DEBUG
            # print(f"[DEBUG] PRED: label={label}, conf={confidence:.2f}")


            if confidence < self.CONFIDENCE_THRESHOLD:
                student_id = self.known_faces.get(label)
                return student_id, confidence
            else:
                return None, confidence
                
        except Exception as e:
            return None, 0
    
    def get_facial_landmarks(self, gray_frame, face):
        """Extract facial landmarks for head pose"""
        x, y, w, h = face
        roi = gray_frame[y:y+h, x:x+w]
        
        # Enhanced preprocessing
        roi = cv2.equalizeHist(roi)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 3, minSize=(30, 30)) # FIX THIS LINE 
        
        landmarks = []
        
        # Nose tip
        nose_x = x + w//2
        nose_y = y + h//3
        landmarks.append((nose_x, nose_y))
        
        # Chin
        chin_x = x + w//2
        chin_y = y + h - h//8
        landmarks.append((chin_x, chin_y))
        
        # Eye corners
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            
            left_eye = eyes[0]
            left_corner_x = x + left_eye[0] - 5
            left_corner_y = y + left_eye[1] + left_eye[3]//2
            landmarks.append((left_corner_x, left_corner_y))
            
            right_eye = eyes[-1]
            right_corner_x = x + right_eye[0] + right_eye[2] + 5
            right_corner_y = y + right_eye[1] + right_eye[3]//2
            landmarks.append((right_corner_x, right_corner_y))
        else:
            # Fallback positions
            left_eye_x = x + w//4
            right_eye_x = x + 3*w//4
            eye_y = y + h//3
            landmarks.append((left_eye_x, eye_y))
            landmarks.append((right_eye_x, eye_y))
        
        # Mouth corners
        mouth_y = y + 2*h//3
        left_mouth_x = x + w//3
        right_mouth_x = x + 2*w//3
        landmarks.append((left_mouth_x, mouth_y))
        landmarks.append((right_mouth_x, mouth_y))
        
        return np.array(landmarks, dtype="double")
    
    def calculate_head_pose(self, landmarks):
        """Calculate head pose from landmarks"""
        if len(landmarks) < 6:
            return None, None, None
        
        try:
            image_points = landmarks.astype(np.float64)
            model_points = self.model_points.astype(np.float64)
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None, None, None
            
            # Convert to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            
            if sy > 1e-6:
                pitch = np.arctan2(-rotation_matrix[2,0], sy) * 180.0 / np.pi
                yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180.0 / np.pi
                roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) * 180.0 / np.pi
            else:
                pitch = np.arctan2(-rotation_matrix[2,0], sy) * 180.0 / np.pi
                yaw = 0
                roll = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1]) * 180.0 / np.pi
            
            # Normalize angles
            pitch = max(-90, min(90, pitch))
            yaw = max(-180, min(180, yaw))
            roll = max(-180, min(180, roll))
            
            return pitch, yaw, roll
            
        except Exception as e:
            return None, None, None
    
    def detect_motion(self, current_frame):
        """Detect motion between frames"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False
        
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_current, gray_prev)
        blur = cv2.GaussianBlur(diff, (5,5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        
        motion_area = cv2.countNonZero(thresh)
        self.prev_frame = current_frame.copy()
        
        return motion_area > self.motion_threshold
    
    def analyze_head_pose_behavior(self, pitch, yaw, roll):
        """Original head pose behavior analysis with improved thresholds"""
    
        # HEAD TILT DETECTION (40° threshold - more realistic)
        if abs(roll) > 40:
            return "UNUSUAL_HEAD_TILT", f"Head tilted {roll:.1f}°"
    
        # LOOKING AWAY DETECTION (50° threshold - more realistic)
        if abs(yaw) > 50:
            direction = "left" if yaw < 0 else "right"
            return "LOOKING_AWAY", f"Looking {direction} {abs(yaw):.1f}°"
    
        # HEAD DOWN DETECTION (35° threshold - more realistic)
        if pitch < -35:
            return "HEAD_DOWN_SUSPICIOUS", f"Head down {pitch:.1f}°"
    
        # NO SUSPICIOUS BEHAVIOR DETECTED
        return None, ""

    


    def update_smart_threshold(self, behavior_type, current_time, intensity):
        """Update smart threshold tracking for behavior"""
        threshold_data = self.smart_thresholds[behavior_type]

        # Check if this is a continuation of previous detection
        if current_time - threshold_data['last_detection_time'] < 0.5:  # Within 0.5s
            threshold_data['current_duration'] += (current_time - threshold_data['last_detection_time'])
            threshold_data['consecutive_detections'] += 1
        else:
            # Reset if gap too large
            threshold_data['current_duration'] = 0.1  # Start fresh
            threshold_data['consecutive_detections'] = 1

        threshold_data['last_detection_time'] = current_time



    def reset_smart_threshold(self, behavior_type, current_time):
        """Reset threshold tracking when behavior stops"""
        if hasattr(self, 'smart_thresholds') and behavior_type in self.smart_thresholds:
            threshold_data = self.smart_thresholds[behavior_type]
            if current_time - threshold_data.get('last_detection_time', 0) > 1.0:
                threshold_data['current_duration'] = max(0, threshold_data.get('current_duration', 0) - 0.5)
                threshold_data['consecutive_detections'] = 0
                if threshold_data['current_duration'] == 0:
                    threshold_data['last_detection_time'] = 0

    def get_alert_level(self, behavior_type):
        """Determine alert level based on duration"""
        if not hasattr(self, 'smart_thresholds') or behavior_type not in self.smart_thresholds:
            return 'NORMAL'
    
        threshold_data = self.smart_thresholds[behavior_type]
        duration = threshold_data.get('current_duration', 0)
    
        if duration >= threshold_data.get('alert_threshold', 4.0):
            return 'RED'
        elif duration >= threshold_data.get('warning_threshold', 2.0):
            return 'ORANGE'
        else:
            return 'NORMAL'

    def create_smart_alert(self, alert_type, details, alert_level):
        """Create smart alert with context"""
        enhanced_details = f"{details} | Level: {alert_level}"
        return alert_type, enhanced_details, alert_level



    def get_alert_level(self, behavior_type):
        """Determine alert level based on duration and intensity"""
        threshold_data = self.smart_thresholds[behavior_type]
        duration = threshold_data['current_duration']

        if duration >= threshold_data['alert_threshold']:
            return 'RED'
        elif duration >= threshold_data['warning_threshold']:
            return 'ORANGE' 
        elif duration >= threshold_data['duration_threshold'] * 0.5:
            return 'YELLOW'
        else:
            return 'NORMAL'
        

    def create_smart_alert(self, alert_type, details, alert_level):
        """Create smart alert with context and level"""
        # Add behavioral context to alert
        context_info = self.get_behavioral_context()
        enhanced_details = f"{details} | Context: {context_info} | Level: {alert_level}"

        return alert_type, enhanced_details, alert_level

    def get_behavioral_context(self):
        """Get current behavioral context for alerts"""
        context = self.behavioral_context
        phase = context['exam_phase']
        intensity = context['activity_intensity']

        if phase == 'reading' and intensity < 30:
            return "Reading phase - low movement expected"
        elif phase == 'writing' and intensity > 50:
            return "Writing phase - moderate movement normal"
        elif intensity > 80:
            return "High activity detected - possible stress"
        else:
            return "Normal exam behavior"
        

    def detect_behavioral_context(self, faces, movement_detected):
        """Detect current exam phase and behavioral context"""
        current_time = time.time()

        # Simple phase detection based on movement patterns
        if movement_detected and len(faces) == 1:
            self.behavioral_context['activity_intensity'] = min(100, 
                self.behavioral_context['activity_intensity'] + 2)
        else:
            self.behavioral_context['activity_intensity'] = max(0, 
                self.behavioral_context['activity_intensity'] - 1)

        # Phase detection logic
        intensity = self.behavioral_context['activity_intensity']
        if intensity > 60:
            new_phase = 'writing'
        elif intensity < 20:
            new_phase = 'reading'
        else:
            new_phase = 'thinking'

        # Update phase if changed
        if new_phase != self.behavioral_context['exam_phase']:
            self.behavioral_context['exam_phase'] = new_phase
            self.behavioral_context['phase_start_time'] = current_time


    def calculate_movement_intensity(self, faces, frame):
        """Calculate movement intensity for behavioral analysis"""
        if len(faces) != 1:
            return 0

        current_time = time.time()
        x, y, w, h = faces[0]
        head_center = (x + w//2, y + h//2)

        # Add to movement history
        self.movement_history.append({
            'position': head_center,
            'timestamp': current_time
        })

        # Keep last 10 seconds of data
        self.movement_history = [
            pos for pos in self.movement_history
            if current_time - pos['timestamp'] < 10.0
        ]

        if len(self.movement_history) < 5:
            return 0

    # Calculate movement intensity
        positions = [pos['position'] for pos in self.movement_history[-10:]]
        if len(positions) < 2:
            return 0

        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = (dx*dx + dy*dy) ** 0.5
            total_movement += movement

        return total_movement / len(positions)
    
    def smart_no_face_detection(self, faces, frame):
        """Smart no-face detection with time thresholds"""
        current_time = time.time()

        if len(faces) == 0:
            self.update_smart_threshold('NO_FACE', current_time, 1.0)
            alert_level = self.get_alert_level('NO_FACE')

            if alert_level == 'RED':
                return 'STUDENT_MISSING', f"No face detected for {self.smart_thresholds['NO_FACE']['current_duration']:.1f}s", alert_level
            elif alert_level == 'ORANGE':
                return 'STUDENT_ABSENT_WARNING', f"Face missing for {self.smart_thresholds['NO_FACE']['current_duration']:.1f}s", alert_level
        else:
            self.reset_smart_threshold('NO_FACE', current_time)

        return None, "", "NORMAL"
    


    def draw_smart_alerts_on_frame(self, frame, alert_level):
        """Draw smart alert indicators on frame"""
        if alert_level == 'NORMAL':
            return

        alert_config = self.alert_levels[alert_level]
        color = alert_config['color']
        text = alert_config['text']

        # Draw alert level indicator
        #cv2.putText(frame, f"ALERT LEVEL: {text}", (10, frame.shape[0] - 60), 
          #         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Draw border based on alert level
        thickness = 5 if alert_level == 'RED' else 3
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, thickness)

        # Show threshold progress bars
        y_offset = frame.shape[0] - 120
        for i, (behavior, data) in enumerate(self.smart_thresholds.items()):
            if data['current_duration'] > 0:
                progress = min(1.0, data['current_duration'] / data['alert_threshold'])
                bar_width = int(200 * progress)
                bar_color = color if progress > 0.8 else (0, 255, 255)

                cv2.rectangle(frame, (10, y_offset + i*20), (210, y_offset + i*20 + 15), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, y_offset + i*20), (10 + bar_width, y_offset + i*20 + 15), bar_color, -1)
                cv2.putText(frame, behavior[:8], (220, y_offset + i*20 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        
    

    def log_suspicious_activity(self, activity_type, details=""):
        """Log suspicious activity with working cooldown system"""
        current_time = time.time()
        now = datetime.now()
    
        # Initialize last_alert_time if needed
        if not hasattr(self, 'last_alert_time'):
            self.last_alert_time = {}
    
        # WORKING COOLDOWN CHECK (FIXED)
        cooldown = 15  # 15 seconds cooldown for all alerts
        if activity_type in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[activity_type]
            if time_since_last < cooldown:
                return  # Skip alert due to cooldown
    
        # Create ISO timestamp
        iso_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        session_time = current_time - self.session_start_time if self.session_start_time > 0 else 0
    
        # Create activity object
        activity = {
            'type': activity_type,
            'details': details,
            'timestamp': iso_timestamp,
            'session_time': f"{session_time:.1f}s",
            'student_id': self.current_student,
            'student_name': self.current_student_info.get('name') if self.current_student_info else 'Unknown',
            'alert_id': f"{self.current_student}_{int(current_time)}"
        }
    
        # Update last alert time BEFORE sending (important!)
        self.last_alert_time[activity_type] = current_time
    
        # Add to local list
        if not hasattr(self, 'suspicious_activities'):
            self.suspicious_activities = []
        self.suspicious_activities.append(activity)
    
        # Terminal output
        print(f"🚨 {activity_type}: {details} at {now.strftime('%H:%M:%S')}")
    
        # Send to Firebase
        try:
            alert_url = f"{self.firebase_url}/exam_alerts.json"
            response = requests.post(alert_url, json=activity, timeout=5)
        
            if response.status_code == 200:
                print(f"✅ Firebase alert sent: {activity_type}")
            else:
                print(f"⚠️ Firebase error: {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ Firebase error: {str(e)}")


        
    
    def process_face_recognition(self, faces, gray, frame):
        """Process face recognition and suspicious activity detection"""
        # print(f"DEBUG: process_face_recognition called, Faces detected: {len(faces)}")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Skip monitoring during toilet break
        if self.toilet_break_gesture['break_active']:
            cv2.putText(frame, "MONITORING SUSPENDED - TOILET BREAK", (10, 200), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return

        
        # No face detection
        if len(faces) == 0:
            self.no_face_count += 1
            cv2.putText(frame, f"No Face: {self.no_face_count}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            if self.no_face_count > self.NO_FACE_THRESHOLD:
                '''cv2.putText(frame, "⚠️ STUDENT MISSING!", (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)'''
                if self.no_face_count == self.NO_FACE_THRESHOLD + 1:
                    self.log_suspicious_activity("STUDENT_MISSING", f"No face for {self.NO_FACE_THRESHOLD} frames")
        
        # Multiple faces
        elif len(faces) > 1:
            self.no_face_count = 0
            self.multiple_face_count += 1
            
            '''cv2.putText(frame, "⚠️ MULTIPLE PEOPLE!", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)'''
            
            if self.multiple_face_count == self.MULTIPLE_FACE_THRESHOLD:
                self.log_suspicious_activity("MULTIPLE_PEOPLE", f"{len(faces)} faces detected")
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Single face - ideal case
        elif len(faces) == 1:
            self.no_face_count = 0
            self.multiple_face_count = max(0, self.multiple_face_count - 1)
            x, y, w, h = faces[0]
            
            # Face recognition
            face_roi = gray[y:y+h, x:x+w]
            student_id, confidence = self.recognize_face(face_roi)
            
            if student_id and student_id == self.current_student:
                # Correct student present
                self.unknown_face_count = 0
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                student_name = self.current_student_info.get('name', 'Unknown')
                cv2.putText(frame, f"{student_name}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {self.current_student_info.get('roll_number', 'Unknown')}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # HEAD POSE ANALYSIS
                landmarks = self.get_facial_landmarks(gray, faces[0])
                if landmarks is not None and len(landmarks) >= 6:
                    pitch, yaw, roll = self.calculate_head_pose(landmarks)
                    
                    if pitch is not None:
                        # Analyze cheating behavior
                        activity_type, details, = self.analyze_head_pose_behavior(pitch, yaw, roll)
                        alert_level = "RED" if activity_type else "NORMAL"
                        
                        if activity_type and alert_level in ['ORANGE', 'RED']:
                            self.log_suspicious_activity(activity_type, details)

                        self.draw_smart_alerts_on_frame(frame, alert_level)
                        
                        # Display head pose info
                        self.draw_enhanced_head_pose_info(frame, pitch, yaw, roll, alert_level)
            
            elif student_id and student_id != self.current_student:
                # Different registered student
                other_student_info = self.student_database.get(student_id, {})
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                '''cv2.putText(frame, "⚠️ WRONG STUDENT!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)'''
                cv2.putText(frame, f"Expected: {self.current_student_info.get('name', 'Unknown')}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                cv2.putText(frame, f"Found: {other_student_info.get('name', 'Unknown')}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                
                self.log_suspicious_activity("WRONG_STUDENT", 
                    f"Expected {self.current_student_info.get('name')}, found {other_student_info.get('name')}")
            
            else:
                # Unknown face
                self.unknown_face_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "UNKNOWN PERSON", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if self.unknown_face_count > self.UNKNOWN_FACE_THRESHOLD:
                    cv2.putText(frame, "⚠️ UNREGISTERED PERSON!", (10, 290), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if self.unknown_face_count == self.UNKNOWN_FACE_THRESHOLD + 1:
                        self.log_suspicious_activity("UNKNOWN_PERSON", "Unregistered face detected")
    

    def draw_improved_overlay(self, frame, alert_level="NORMAL", student_name="Unknown", 
                         confidence=0, alert_type=None, head_pose=None):
        """Draw improved overlay with better positioning and readability"""
        h, w = frame.shape[:2]
    
        # Color scheme based on alert level
        colors = {
            'NORMAL': (0, 255, 0),    # Green
            'YELLOW': (0, 255, 255),  # Yellow  
            'ORANGE': (0, 165, 255),  # Orange
            'RED': (0, 0, 255)        # Red
        }
    
        alert_color = colors.get(alert_level, (255, 255, 255))
    
        # Top status bar with background
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 40), alert_color, 2)
    
        # Student info and confidence
        cv2.putText(frame, f"Student: {student_name}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
        if confidence > 0:
            cv2.putText(frame, f"Conf: {confidence:.1f}%", (w - 120, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        # Alert status (only when needed)
        if alert_level != "NORMAL":
            cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, h-50), (w, h), alert_color, 2)
        
            status_text = f"ALERT: {alert_type}" if alert_type else f"STATUS: {alert_level}"
            cv2.putText(frame, status_text, (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

    
    def draw_head_pose_info(self, frame, pitch, yaw, roll):
        """Draw head pose information on frame"""
        # Color coding based on thresholds
        pitch_color = (0, 0, 255) if pitch < -25 else (0, 255, 0)
        yaw_color = (0, 0, 255) if abs(yaw) > 40 else (0, 255, 0)
        
        cv2.putText(frame, f"Pitch: {pitch:.1f}°", (frame.shape[1]-200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 1)
        cv2.putText(frame, f"Yaw: {yaw:.1f}°", (frame.shape[1]-200, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 1)
        cv2.putText(frame, f"Roll: {roll:.1f}°", (frame.shape[1]-200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Alert indicators
        if self.head_down_count > 30:
            cv2.putText(frame, f"⚠️ HEAD DOWN: {self.head_down_count/30:.1f}s", 
                       (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if self.looking_side_count > 20:
            cv2.putText(frame, f"⚠️ LOOKING AROUND: {self.looking_side_count/30:.1f}s", 
                       (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    def start_monitoring_session(self):
        motion_detected = False
        movement_intensity = 0
        alert_level = "NORMAL"
        no_face_alert_type = None
        no_face_details = ""
        no_face_level = "NORMAL"
        """Start the complete monitoring session"""
        print(f"\n🚀 MONITORING ACTIVE")
        print(f"👤 Student: {self.current_student_info.get('name')} ({self.current_student})")
        print(f"🔢 Roll: {self.current_student_info.get('roll_number')}")
        print("📋 Controls: 'q' to quit, 's' to save report, 'r' to restart, 'e' to end session")
        print("="*80)
        
        frame_count = 0
        
        while self.exam_session_active:
            ret, frame = self.cap.read()
            # 🤚 PHASE 2: GESTURE DETECTION
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = self.hands.process(rgb_frame)

            # Check for toilet break gesture
            #gesture_detected = self.detect_toilet_break_gesture(frame, hands_results)

            # 🔥 PHASE 2.1: Advanced Gestures
            toilet_gesture = self.detect_toilet_break_gesture(frame, hands_results)
            advanced_gesture = self.detect_advanced_gestures(frame, hands_results)

            gesture_detected = toilet_gesture or advanced_gesture

            # Check toilet break status
            break_active = self.check_toilet_break_status()

            # Draw gesture overlay
            self.draw_gesture_overlay(frame, hands_results)


            if not ret:
                continue
            
            frame_count += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05, # Tweak this value
                minNeighbors=5,   # Tweak this value
                minSize=(100, 100),
                maxSize=(400, 400)
            )

            # SMART NO-FACE DETECTION
        #no_face_alert_type, no_face_details, no_face_level = self.smart_no_face_detection(faces, frame)
            no_face_alert_type = None
            no_face_details = ''
            no_face_level = 'NORMAL'

            if no_face_alert_type and no_face_level in ['ORANGE', 'RED']:
                self.log_suspicious_activity(no_face_alert_type, no_face_details)

            # DETECT BEHAVIORAL CONTEXT
            motion_detected = False
            self.detect_behavioral_context(faces, motion_detected)

            # MOVEMENT INTENSITY ANALYSIS  
            if len(faces) == 1:
                movement_intensity = self.calculate_movement_intensity(faces, frame)

            else:
                movement_intensity = 0 
            if movement_intensity > self.smart_thresholds['RESTLESS_BEHAVIOR']['movement_threshold']:
                current_time = time.time()
                self.update_smart_threshold('RESTLESS_BEHAVIOR', current_time, movement_intensity)
                alert_level = self.get_alert_level('RESTLESS_BEHAVIOR')

                if alert_level in ['ORANGE', 'RED']:
                    self.log_suspicious_activity('RESTLESS_BEHAVIOR', 
                        f"High movement intensity {movement_intensity:.1f} for {self.smart_thresholds['RESTLESS_BEHAVIOR']['current_duration']:.1f}s")

            
            # Motion detection
            motion_detected = self.detect_motion(frame)
            
            # 📊 MAIN HEADER INFO
            session_time = time.time() - self.session_start_time
            '''cv2.putText(frame, f"🎯 EXAM SESSION: {int(session_time//60)}m {int(session_time%60)}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)'''
            cv2.putText(frame, f"Student: {self.current_student_info.get('name', 'Unknown')}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {self.current_student_info.get('roll_number', 'Unknown')}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(faces)} | Motion: {'YES' if motion_detected else 'NO'}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"🚨 Total Alerts: {len(self.suspicious_activities)}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 🧠 PROCESS FACE RECOGNITION + HEAD POSE
            self.process_face_recognition(faces, gray, frame)

            # After face detection
            #self.process_face_recognition(faces, gray, frame)
            #self.detect_behavioral_cheating(faces, frame)  # ← Add this
            
            # 📺 Display frame
            cv2.imshow("🧠 INTEGRATED PROCTORING SYSTEM - LIVE MONITORING", frame)
            
            # ⌨️ Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 System shutdown requested")
                break
            elif key == ord('s'):
                self.save_exam_report()
            elif key == ord('r'):
                self.restart_session()
                break
            elif key == ord('e'):
                print("📝 Ending current exam session...")
                self.end_exam_session()
                break
        
        # Auto-save and cleanup
        if self.exam_session_active:
            self.save_exam_report()

    def draw_enhanced_head_pose_info(self, frame, pitch, yaw, roll, alert_level):
        """Draw enhanced head pose information with smart indicators"""
        # Get alert colors
        alert_config = self.alert_levels.get(alert_level, self.alert_levels['NORMAL'])
        alert_color = alert_config['color']

        # Color coding based on smart thresholds
        pitch_color = alert_color if abs(pitch) > 25 else (0, 255, 0)
        yaw_color = alert_color if abs(yaw) > 40 else (0, 255, 0)

        # Enhanced display with threshold indicators
        '''cv2.putText(frame, f"Pitch: {pitch:.1f}° ({'ALERT' if abs(pitch) > 25 else 'OK'})", 
                   (frame.shape[1]-250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 1)
        cv2.putText(frame, f"Yaw: {yaw:.1f}° ({'ALERT' if abs(yaw) > 40 else 'OK'})", 
                   (frame.shape[1]-250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 1)'''
        cv2.putText(frame, f"Roll: {roll:.1f}° ({'ALERT' if abs(roll) > 25 else 'OK'})", 
                   (frame.shape[1]-250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   alert_color if abs(roll) > 25 else (255, 255, 255), 1)

        # Show behavioral context
        context = self.behavioral_context
        cv2.putText(frame, f"Phase: {context['exam_phase'].upper()}", 
                   (frame.shape[1]-250, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Activity: {context['activity_intensity']:.0f}%", 
                   (frame.shape[1]-250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Show active threshold timers
        active_alerts = []
        for behavior, data in self.smart_thresholds.items():
            if data['current_duration'] > 0.5:
                active_alerts.append(f"{behavior}: {data['current_duration']:.1f}s")

        if active_alerts:
            y_start = 180
            for i, alert_text in enumerate(active_alerts[:3]):  # Show max 3
                cv2.putText(frame, alert_text, (frame.shape[1]-250, y_start + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
    
    def end_exam_session(self):
        """End current exam session and return to RFID wait"""
        if self.exam_session_active:
            self.save_exam_report()
            
            print(f"\n📋 EXAM SESSION ENDED")
            print(f"👤 Student: {self.current_student_info.get('name')} ({self.current_student})")
            print(f"⏰ Duration: {int((time.time() - self.session_start_time)//60)}m {int((time.time() - self.session_start_time)%60)}s")
            print(f"🚨 Total Alerts: {len(self.suspicious_activities)}")
        
        # Reset session
        self.exam_session_active = False
        self.current_student = None
        self.current_student_info = None
        self.session_start_time = 0
        self.suspicious_activities = []
        
        # Reset counters
        self.no_face_count = 0
        self.unknown_face_count = 0
        self.head_down_count = 0
        self.looking_side_count = 0
        self.multiple_face_count = 0
        
        print("\n🔄 Returning to RFID scan mode...")
    
    def restart_session(self):
        """Restart exam session"""
        self.end_exam_session()
        # Will automatically go back to wait_for_rfid_scan in main loop
    
    def save_exam_report(self):
        """Save comprehensive exam report"""
        if self.session_start_time == 0:
            print("⚠️ No active session to save")
            return
        
        session_duration = time.time() - self.session_start_time
        
        # Calculate summary statistics
        alert_types = {}
        for activity in self.suspicious_activities:
            alert_type = activity['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        session_id = f"{self.current_student}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_generated = datetime.now().isoformat()
        
        report = {
            'session_id': session_id,
            'report_generated': report_generated,
            'exam_session': {
                'student_id': self.current_student,
                'student_name': self.current_student_info.get('name', 'Unknown') if self.current_student_info else 'Unknown',
                'roll_number': self.current_student_info.get('roll_number', 'Unknown') if self.current_student_info else 'Unknown',
                'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'duration_seconds': session_duration,
                'duration_formatted': f"{int(session_duration//60)}m {int(session_duration%60)}s",
                'end_time': datetime.now().isoformat()
            },
            'monitoring_summary': {
                'total_alerts': len(self.suspicious_activities),
                'alert_breakdown': alert_types,
                'monitoring_components': [
                    'Face Recognition',
                    'Head Pose Detection', 
                    'Motion Detection',
                    'Multi-person Detection',
                    'RFID Integration'
                ]
            },
            'detection_thresholds': {
                'no_face_threshold_frames': self.NO_FACE_THRESHOLD,
                'head_down_threshold_frames': self.HEAD_DOWN_THRESHOLD,
                'looking_side_threshold_frames': self.LOOKING_SIDE_THRESHOLD,
                'unknown_face_threshold_frames': self.UNKNOWN_FACE_THRESHOLD,
                'confidence_threshold': self.CONFIDENCE_THRESHOLD
            },
            'detailed_activities': self.suspicious_activities,
            'system_info': {
                'firebase_integration': True,
                'rfid_integration': True,
                'face_recognition_enabled': len(self.known_faces) > 0,
                'head_pose_detection_enabled': True,
                'total_registered_students': len(self.student_database)
            }
        }
        
        # Save local report
        filename = f"exam_reports/exam_report_{session_id}.json"
        os.makedirs("exam_reports", exist_ok=True)
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #filename = f"exam_reports/exam_report_{self.current_student}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Send to Firebase
        try:
            report_url = f"{self.firebase_url}/monitoring_reports/{session_id}.json" # type: ignore
            requests.put(report_url, json=report, timeout=10)
            print("🔥 Report uploaded to Firebase!")
        except Exception as e:
            print(f"⚠️ Firebase upload failed: {e}")
        
        # Display summary
        print("\n" + "="*60)
        print("📄 EXAM REPORT GENERATED")
        print("="*60)
        print(f"👤 Student: {self.current_student_info.get('name', 'Unknown')}")
        print(f"🔢 Roll: {self.current_student_info.get('roll_number', 'Unknown')}")
        print(f"📱 Card: {self.current_student}")
        print(f"⏰ Duration: {report['exam_session']['duration_formatted']}")
        print(f"🚨 Total Alerts: {len(self.suspicious_activities)}")
        
        if alert_types:
            print("📊 Alert Breakdown:")
            for alert_type, count in alert_types.items():
                print(f"   • {alert_type}: {count}")
        
        print(f"📁 Saved: {filename}")
        print("="*60)

    
# ADD THIS TO YOUR PYTHON SCRIPT - Enhanced Report Generation System

    def generate_comprehensive_report(self):
        """Generate detailed exam session report with analytics"""
        if not self.current_student or not self.suspicious_activities:
            return None

        end_time = time.time()
        session_duration = end_time - self.session_start_time

    # Calculate advanced analytics
        alert_types = {}
        time_distribution = []
        risk_score = 0

        for activity in self.suspicious_activities:
            alert_type = activity['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        # Calculate risk scores
            if 'WRONG_STUDENT' in alert_type:
                risk_score += 10
            elif 'STUDENT_MISSING' in alert_type:
                risk_score += 8
            elif 'MULTIPLE_FACES' in alert_type:
                risk_score += 7
            elif 'UNUSUAL_HEAD_TILT' in alert_type:
                risk_score += 3
            elif 'RESTLESS_BEHAVIOR' in alert_type:
                risk_score += 2

    # Determine risk level
        if risk_score >= 30:
            risk_level = "HIGH RISK"
            risk_color = "danger"
        elif risk_score >= 15:
            risk_level = "MEDIUM RISK"
            risk_color = "warning"
        elif risk_score >= 5:
            risk_level = "LOW RISK" 
            risk_color = "info"
        else:
            risk_level = "MINIMAL RISK"
            risk_color = "success"

    # Generate comprehensive report
        report = {
            'session_id': f"SESSION_{int(self.session_start_time)}",
            'report_generated': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            'exam_session': {
                'student_id': self.current_student,
                'student_name': self.current_student_info.get('name', 'Unknown'),
                'start_time': datetime.fromtimestamp(self.session_start_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                'end_time': datetime.fromtimestamp(end_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                'duration_seconds': int(session_duration),
                'duration_formatted': f"{int(session_duration//60)}m {int(session_duration%60)}s",
                'session_status': 'COMPLETED'
            },
            'monitoring_summary': {
                'total_alerts': len(self.suspicious_activities),
                'unique_alert_types': len(alert_types),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'alert_frequency': round(len(self.suspicious_activities) / (session_duration/60), 2),  # alerts per minute
                'monitoring_efficiency': 'HIGH'  # You can calculate this based on detection accuracy
            },
            'alert_breakdown': alert_types,
            'suspicious_activities': self.suspicious_activities,
            'behavioral_analysis': {
                'head_movement_alerts': alert_types.get('UNUSUAL_HEAD_TILT', 0),
                'identity_verification_issues': alert_types.get('WRONG_STUDENT', 0) + alert_types.get('UNKNOWN_PERSON', 0),
                'attention_span_issues': alert_types.get('RESTLESS_BEHAVIOR', 0) + alert_types.get('LOOKING_AWAY', 0),
                'presence_issues': alert_types.get('STUDENT_MISSING', 0) + alert_types.get('NO_FACE_DETECTED', 0)
            },
            'recommendations': self.generate_recommendations(alert_types, risk_score),
            'ai_insights': self.generate_ai_insights(alert_types, session_duration)
        }

    # Send to Firebase
        try:
            reports_url = f"{self.firebase_url}/monitoring_reports.json"
            response = requests.post(reports_url, json=report, timeout=5)
            if response.status_code == 200:
                print(f"✅ Comprehensive report generated: {report['session_id']}")
                return report
        except Exception as e:
            print(f"⚠️ Report generation failed: {e}")

        return report

    def generate_recommendations(self, alert_types, risk_score):
        """Generate AI-powered recommendations based on monitoring data"""
        recommendations = []

        if alert_types.get('UNUSUAL_HEAD_TILT', 0) > 5:
            recommendations.append({
                'type': 'BEHAVIOR',
                'priority': 'MEDIUM',
                'message': 'Consider adjusting camera angle or student seating position',
                'action': 'Camera calibration recommended'
            })

        if alert_types.get('WRONG_STUDENT', 0) > 0:
            recommendations.append({
                'type': 'IDENTITY',
                'priority': 'HIGH', 
                'message': 'Identity verification failed - manual review required',
                'action': 'Immediate supervisor intervention needed'
            })

        if alert_types.get('STUDENT_MISSING', 0) > 3:
            recommendations.append({
                'type': 'PRESENCE',
                'priority': 'HIGH',
                'message': 'Frequent absence detected - possible unauthorized break',
                'action': 'Review session recording and student conduct'
            })

        if risk_score > 20:
            recommendations.append({
                'type': 'OVERALL',
                'priority': 'HIGH',
                'message': 'High-risk session detected - comprehensive review recommended',
                'action': 'Flag for detailed examination and possible retest'
            })

        if not recommendations:
            recommendations.append({
                'type': 'POSITIVE',
                'priority': 'INFO',
                'message': 'Excellent exam conduct maintained throughout session',
                'action': 'No further action required'
            })

        return recommendations

    def generate_ai_insights(self, alert_types, session_duration):
        """Generate AI-powered insights and patterns"""
        insights = []

        total_alerts = sum(alert_types.values())

        if total_alerts == 0:
            insights.append("🎯 Perfect monitoring session with zero suspicious activities detected")
            insights.append("📊 Student demonstrated excellent exam conduct and focus")
        else:
        # Pattern analysis
            if alert_types.get('UNUSUAL_HEAD_TILT', 0) > total_alerts * 0.7:
                insights.append("📐 Primary concern: Head positioning - may indicate reading difficulty or screen visibility issues")

            if alert_types.get('RESTLESS_BEHAVIOR', 0) > 3:
                insights.append("⚡ Behavioral pattern: High movement frequency detected - possible stress or anxiety indicators")

        # Time-based insights
            if session_duration > 3600:  # More than 1 hour
                insights.append(f"⏱️ Extended session duration ({int(session_duration/60)} minutes) - fatigue may have contributed to behavioral changes")

        # Frequency analysis
            alert_rate = total_alerts / (session_duration / 60)  # per minute
            if alert_rate > 1:
                insights.append(f"📈 High alert frequency ({alert_rate:.1f}/min) - suggests challenging exam conditions or technical issues")
            elif alert_rate < 0.2:
                insights.append(f"📉 Low alert frequency ({alert_rate:.1f}/min) - indicates stable and focused exam performance")

        return insights

# Call this function when stopping monitoring
    def stop_monitoring_and_generate_report(self):
        """Stop monitoring and generate comprehensive report"""
        if self.monitoring_active:
            self.monitoring_active = False

        # Generate comprehensive report
            report = self.generate_comprehensive_report()

            if report:
                print(f"\n📊 EXAM SESSION COMPLETED")
                print(f"👤 Student: {report['exam_session']['student_name']}")
                print(f"⏱️ Duration: {report['exam_session']['duration_formatted']}")
                print(f"🚨 Total Alerts: {report['monitoring_summary']['total_alerts']}")
                print(f"⚠️ Risk Level: {report['monitoring_summary']['risk_level']}")
                print(f"✅ Report ID: {report['session_id']}")

        # Cleanup
            self.current_student = None
            self.current_student_info = {}
            self.suspicious_activities = []
            self.session_start_time = 0

            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

            return report

    def monitor_arduino_attendance(self):
        """Monitor Arduino attendance with continuous operation"""
        last_record_id = None
        first_run = True

        print("⏰ Arduino monitor starting...")
        print("   Waiting for NEW scans...")

        while True:  # ✅ Loop forever, don't exit!
            try:
                response = requests.get(
                    f"{self.firebase_url}/attendance_records.json",
                    timeout=5
                )

                if response.status_code == 200:
                    data = response.json()

                    if data:
                        latest_id = list(data.keys())[-1]

                        if first_run:
                            last_record_id = latest_id
                            first_run = False
                            print(f"📋 Initialized. Ignoring existing records.")
                            print(f"   Waiting for NEW Arduino scans...")
                            continue

                        if latest_id != last_record_id:
                            last_record_id = latest_id
                            record_data = data[latest_id]

                            card_id = record_data.get('student_id', 'Unknown')
                            fingerprint_id = record_data.get('fingerprint_id', 0)
                            confidence = record_data.get('confidence', 0)
                            status = record_data.get('status', 'unknown')

                            print("\n" + "="*60)
                            print("🎯 NEW ARDUINO ATTENDANCE DETECTED!")
                            print("="*60)
                            print(f"📱 Card ID: {card_id}")
                            print(f"👆 Fingerprint ID: {fingerprint_id}")
                            print(f"📊 Confidence: {confidence}")
                            print(f"✅ Status: {status}")
                            print("="*60)

                            if status == "success":
                                if card_id in self.student_database:
                                    # EXISTING STUDENT
                                    student_info = self.student_database[card_id]

                                    print(f"\n👤 STUDENT FOUND!")
                                    print(f"Name: {student_info['name']}")
                                    print(f"Roll: {student_info.get('roll_number', 'N/A')}")
                                    print(f"")
                                    print(f"✅ Step 1/3: Card verified")
                                    print(f"✅ Step 2/3: Fingerprint verified")
                                    print(f"")
                                    print(f"📸 Step 3/3: Face verification...")
                                    print(f"")

                                    self.face_verification_in_progress = True

                                    face_verified = self.verify_face_for_attendance(card_id, student_info)

                                    self.face_verification_in_progress = False

                                    if face_verified:
                                        print(f"")
                                        print(f"="*60)
                                        print(f"✅✅✅ THREE-FACTOR AUTHENTICATION SUCCESSFUL! ✅✅✅")
                                        print(f"="*60)
                                        print(f"✅ Card: Verified")
                                        print(f"✅ Fingerprint: Verified")
                                        print(f"✅ Face: Verified")
                                        print(f"="*60)
                                        print(f"")

                                        print("📋 What would you like to do?")
                                        print("1. Start monitoring session")
                                        print("2. Re-register student")
                                        print("3. Cancel")

                                        choice = input("\nEnter choice (1/2/3): ").strip()

                                        if choice == '1':
                                            print("\n🚀 Starting monitoring session...")
                                            self.current_student = card_id
                                            self.current_student_info = student_info
                                            self.exam_session_active = True
                                            self.session_start_time = time.time()
                                            print("✅ Monitoring started!")
                                            return  # ✅ Exit to start monitoring

                                        elif choice == '2':
                                            print("\n🔄 Re-registering student...")
                                            confirm = input(f"⚠️ Delete existing data for {student_info['name']}? (y/n): ")
                                            if confirm.lower() == 'y':
                                                self.register_new_student(card_id)
                                                # ✅ DON'T return - reload database and continue!
                                                self.load_student_database()
                                                print("\n✅ Re-registration complete!")
                                                print("📡 Ready for next scan...")
                                            else:
                                                print("❌ Cancelled")

                                        else:
                                            print("❌ Cancelled")
                                            print("📡 Ready for next scan...")
                                    else:
                                        print(f"")
                                        print(f"❌ FACE VERIFICATION FAILED!")
                                        print(f"Authentication failed - Access denied.")
                                        print(f"📡 Ready for next scan...")

                                else:
                                    # NEW STUDENT
                                    print("\n🆕 NEW STUDENT DETECTED!")
                                    print(f"Card ID: {card_id}")
                                    confirm = input("\nProceed with registration? (y/n): ").strip().lower()

                                    if confirm == 'y':
                                        self.register_new_student(card_id)
                                        # ✅ DON'T return - reload and continue!
                                        self.load_student_database()
                                        print("\n✅ Registration complete!")
                                        print("📡 Ready for next scan...")
                                    else:
                                        print("❌ Cancelled")
                                        print("📡 Ready for next scan...")
                            else:
                                print(f"⚠️ Arduino authentication failed!")
                                print(f"📡 Ready for next scan...")

                time.sleep(2)

            except Exception as e:
                print(f"⚠️ Arduino monitor error: {e}")
                time.sleep(5)

    def verify_face_for_attendance(self, card_id, student_info):
        """
        Optimized face verification with better accuracy
        """
        print("📸 Starting face verification...")
        print("Instructions:")
        print("- Look directly at the camera")
        print("- Face the camera straight")
        print("- Press 'c' to capture (takes 3 samples)")
        print("- Press 'q' to cancel")
        print()

        samples_to_verify = 3  # Take multiple samples for better accuracy
        verification_samples = []

        # ✅ Create window and bring to focus
        cv2.namedWindow('Face Verification', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Verification', 800, 600)

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            # Instructions
            cv2.putText(display_frame, "FACE VERIFICATION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Student: {student_info['name']}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Samples: {len(verification_samples)}/{samples_to_verify}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(120, 120))

            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display_frame, "Face detected - Press 'c'", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(faces) > 1:
                cv2.putText(display_frame, "Multiple faces - adjust position!", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No face detected!", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display_frame, "Press 'q' to cancel", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Verification', display_frame)

            key = cv2.waitKey(50) & 0xFF

            if frame_count % 30 == 0:
                try:
                    cv2.setWindowProperty('Face Verification', cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty('Face Verification', cv2.WND_PROP_TOPMOST, 0)
                except:
                    pass

            if (key == ord('c') or key == ord('C')) and len(faces) == 1:
                # Capture sample
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]

                # Recognize
                recognized_id, confidence = self.recognize_face(face_roi)

                verification_samples.append({
                    'id': recognized_id,
                    'confidence': confidence
                })

                print(f"Sample {len(verification_samples)}/{samples_to_verify}:")
                print(f"  Detected: {recognized_id}")
                print(f"  Confidence: {confidence:.2f}")

                if len(verification_samples) >= samples_to_verify:
                    # Analyze all samples
                    matches = sum(1 for s in verification_samples if s['id'] == card_id)
                    avg_confidence = sum(s['confidence'] for s in verification_samples) / len(verification_samples)

                    print(f"\n📊 Verification Results:")
                    print(f"Matches: {matches}/{samples_to_verify}")
                    print(f"Average Confidence: {avg_confidence:.2f}")

                    # ✅ More lenient: 2 out of 3 samples should match
                    if matches >= 2 and avg_confidence <= self.CONFIDENCE_THRESHOLD:  # Lower threshold = more lenient
                        print(f"✅ Face verified!")
                        cv2.destroyWindow('Face Verification')
                        return True
                    else:
                        print(f"❌ Face verification failed!")
                        print(f"   Expected: {card_id}")
                        print(f"   Got: {matches} matches")
                        cv2.destroyWindow('Face Verification')
                        return False

                # Small delay between samples
                time.sleep(0.5)

            elif key == ord('q') or key == ord('Q'):
                print("❌ Verification cancelled")
                cv2.destroyWindow('Face Verification')
                time.sleep(0.1)
                return False

        cv2.destroyWindow('Face Verification')
        return False

    
    def start_main_system(self):
        """Start the main system loop"""
        print("🎯 INTEGRATED PROCTORING SYSTEM")
        print("="*80)
        print("🔧 System Components:")
        print("   📱 RFID Integration")
        print("   👤 Face Recognition") 
        print("   📐 Head Pose Detection")
        print("   🏃 Motion Detection")
        print("   🚨 Real-time Alert System")
        print("   🔥 Firebase Integration")
        print("   📝 Student Registration")
        print("="*80)
        print(f"📊 Registered Students: {len(self.student_database)}")

        # 🤖 Start Arduino attendance monitor
        arduino_thread = threading.Thread(
            target=self.monitor_arduino_attendance,
            daemon=True
        )
        arduino_thread.start()
        print("🔍 Arduino attendance monitor started...")

        
        # Main system loop
        while True:
            # Wait for RFID scan
            if not self.wait_for_rfid_scan():
                break
            
            # Start monitoring session
            if self.exam_session_active:
                self.start_monitoring_session()
            
            # Check if user wants to continue
            if not self.exam_session_active:
                continue_choice = input("\n🔄 Continue with new scan? (y/n): ").lower()
                if continue_choice != 'y':
                    break
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup system resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🎯 Integrated monitoring system shutdown complete")

# 🚀 MAIN SYSTEM RUNNER
if __name__ == "__main__":
    try:
        print("🚀 STARTING INTEGRATED PROCTORING SYSTEM...")
        print("📋 System will wait for RFID scan to begin...")
        
        system = IntegratedProctoringSystem()
        system.start_main_system()
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()