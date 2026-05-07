import cv2
import numpy as np
import time

class RealAngleDetector:
    def __init__(self):
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Camera with proper setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 3D face model points for accurate pose
        self.model_points = np.array([
            (0.0, 0.0, 0.0),       # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        
        # Camera matrix
        self.cam_matrix = None
        
    def get_landmarks(self, face_rect, frame):
        """Get facial landmarks using face rectangle"""
        x, y, w, h = face_rect
        
        # Approximate landmarks based on face rectangle
        landmarks_2d = np.array([
            [x + w//2, y + h//2],      # Nose
            [x + w//2, y + h - 10],    # Chin
            [x + w//4, y + h//3],      # Left eye
            [x + 3*w//4, y + h//3],    # Right eye
            [x + w//3, y + 2*h//3],    # Left mouth
            [x + 2*w//3, y + 2*h//3]   # Right mouth
        ], dtype=np.float64)
        
        return landmarks_2d
    
    def calculate_head_pose(self, landmarks_2d, frame_shape):
        """Calculate actual head angles"""
        h, w = frame_shape[:2]
        
        # Camera matrix for this frame size
        cam_matrix = np.array([
            [w*0.8, 0, w/2],
            [0, w*0.8, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            landmarks_2d,
            cam_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Get angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]
            
            return pitch, yaw, roll
        
        return 0, 0, 0
    
    def run_angle_test(self):
        print("🎯 Real Angle Detection Test")
        print("=" * 50)
        print("📱 Move your head up/down/left/right")
        print("📊 Watch pitch values change!")
        
        head_down_start = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face
                
                # Get landmarks
                landmarks_2d = self.get_landmarks(face, frame)
                
                # Calculate real angles
                pitch, yaw, roll = self.calculate_head_pose(landmarks_2d, frame.shape)
                
                # Draw face rectangle
                color = (0, 255, 0)
                if pitch < -15:  # Looking down
                    color = (0, 0, 255)
                elif pitch > 15:  # Looking up
                    color = (255, 0, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw landmarks (small circles)
                for point in landmarks_2d:
                    cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 255), -1)
                
                # Display angles
                cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}°", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Head down detection
                if pitch < -20:  # Strict threshold
                    if head_down_start is None:
                        head_down_start = time.time()
                    
                    duration = time.time() - head_down_start
                    cv2.putText(frame, f"HEAD DOWN: {duration:.1f}s", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    head_down_start = None
                
                # Visual guide
                if pitch < -20:
                    cv2.putText(frame, "⚠️ Looking at desk", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                elif -20 <= pitch < -10:
                    cv2.putText(frame, "📖 Reading position", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                else:
                    cv2.putText(frame, "👀 Normal", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Instructions
            cv2.putText(frame, "Pitch <-20° = Head Down", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, "Press Q to quit", (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            cv2.imshow("Real Angle Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealAngleDetector()
    detector.run_angle_test()