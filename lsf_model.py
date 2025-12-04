import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: MediaPipe not found. AI features disabled.")

class LSFDetector:
    def __init__(self):
        if HAS_MEDIAPIPE:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                max_num_hands=1 # Focus on one hand for simple classification
            )
        else:
            self.hands = None

    def process_frame(self, frame):
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if HAS_MEDIAPIPE:
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return results, image
        else:
            # Dummy pass-through
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return None, image

    def draw_landmarks(self, image, results):
        if not HAS_MEDIAPIPE or not results or not results.multi_hand_landmarks:
            return image
            
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        return image

    def extract_keypoints(self, results):
        """
        Extracts 63-dim vector: (x,y,z) for each of 21 landmarks, relative to wrist.
        Matches logic from: https://github.com/Razane1414/Hand-Tracking---Langue-des-signes
        """
        if results is None or not results.multi_hand_landmarks:
            return np.zeros(63)
            
        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Wrist is landmark 0
        wrist = hand_landmarks.landmark[0]
        
        # Landmark 9 is Middle Finger MCP
        middle_mcp = hand_landmarks.landmark[9]

        # Calculate scale (Euclidean distance between wrist and middle_mcp)
        scale = ((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2 + (wrist.z - middle_mcp.z)**2)**0.5
        if scale == 0:
            scale = 1.0

        coords = []
        for lm in hand_landmarks.landmark:
            # Normalize relative to wrist and scale
            coords.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale])
            
        return np.array(coords)
