import cv2
import mediapipe as mp
import math
import pyautogui
import ctypes
import torch

# Global variables for X and Y distances
dx = 0
dy = 0

# Global variables for pinch detection
is_index_pinch = False
is_middle_finger_pinch = False
is_triple_finger_pinch = False

class HandDistanceCalculator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.min_threshold_multiplier = 1.0  # Minimum sensitivity
        self.max_threshold_multiplier = 8.0  # Maximum sensitivity

    def calculate_distance(self, landmarks, frame_shape):
        thumb_tip = (landmarks.landmark[4].x * frame_shape[1], landmarks.landmark[4].y * frame_shape[0])
        palm_base = (landmarks.landmark[17].x * frame_shape[1], landmarks.landmark[17].y * frame_shape[0])
        distance = math.dist(thumb_tip, palm_base)
        return distance

class ThumbTracker:
    def __init__(self):
        self.threshold_multiplier = 1.0  # Initial threshold multiplier
        self.thumb_location = None
        self.prev_thumb_location = None  # Previous thumb location

    def track_thumb(self, frame, landmarks, min_radius=5, max_radius=50):
        for i, landmark in enumerate(landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)

            if i == 4:  # Landmark index 4 corresponds to the tip of the thumb
                # Calculate the radius of the circle based on sensitivity (threshold_multiplier)
                radius = int(max(min_radius, max_radius * self.threshold_multiplier))

                # Draw a circle at the position of the thumb tip
                cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)

                # Store the thumb location
                self.prev_thumb_location = self.thumb_location  # Store previous location
                self.thumb_location = (x, y)

class IndexPinchDetector:
    def __init__(self):
        self.pinch_threshold = 30  # Adjust this threshold as needed

    def detect_index_pinch(self, thumb_location, index_location):
        if thumb_location and index_location:
            distance = math.dist(thumb_location, index_location)
            return distance < self.pinch_threshold
        return False

class MiddleFingerPinchDetector:
    def __init__(self):
        self.pinch_threshold = 30  # Adjust this threshold as needed

    def detect_middle_finger_pinch(self, thumb_location, middle_finger_location):
        if thumb_location and middle_finger_location:
            distance = math.dist(thumb_location, middle_finger_location)
            return distance < self.pinch_threshold
        return False

class TripleFingerPinchDetector:
    def __init__(self):
        self.pinch_threshold = 30  # Adjust this threshold as needed

    def detect_triple_finger_pinch(self, thumb_location, index_location, middle_finger_location):
        if thumb_location and index_location and middle_finger_location:
            distance1 = math.dist(thumb_location, index_location)
            distance2 = math.dist(thumb_location, middle_finger_location)
            return distance1 < self.pinch_threshold and distance2 < self.pinch_threshold
        return False

class CursorSimulator:
    def __init__(self, threshold=10):
        self.threshold = threshold

    def move_cursor(self, dx, dy):
        # Disable PyAutoGUI fail-safe
        pyautogui.FAILSAFE = False
        
        # Apply the threshold to dx and dy
        dx = 0 if abs(dx) < self.threshold else dx
        dy = 0 if abs(dy) < self.threshold else dy
        
        # Negate dy to flip the Y direction if needed
        dy = -dy
        
        # Use pyautogui to move the cursor based on the calculated dx and dy
        pyautogui.move(dx, dy)

def main():
    # Initialize OpenCV webcam capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a different camera index

    distance_calculator = HandDistanceCalculator()
    thumb_tracker = ThumbTracker()
    index_pinch_detector = IndexPinchDetector()
    middle_finger_pinch_detector = MiddleFingerPinchDetector()
    triple_finger_pinch_detector = TripleFingerPinchDetector()
    
    cursor_simulator = CursorSimulator()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hand module
        results = distance_calculator.hands.process(rgb_frame)

        thumb_location = None
        index_location = None
        middle_finger_location = None

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Calculate the estimated distance of the hand from the camera
                hand_distance = distance_calculator.calculate_distance(landmarks, frame.shape)

                # Adjust thumb tracking sensitivity based on hand distance
                normalized_distance = (hand_distance - 100) / 300  # Normalize the distance
                normalized_distance = max(0.0, min(1.0, normalized_distance))  # Clip the value between 0 and 1
                distance_calculator.threshold_multiplier = distance_calculator.min_threshold_multiplier + \
                                                          (distance_calculator.max_threshold_multiplier - distance_calculator.min_threshold_multiplier) * (1 - normalized_distance)
                
                print(distance_calculator.threshold_multiplier)
                # Track the thumb using the ThumbTracker class
                thumb_tracker.track_thumb(frame, landmarks)

                # Get the current and previous thumb locations
                thumb_location = thumb_tracker.thumb_location
                prev_thumb_location = thumb_tracker.prev_thumb_location

                # Calculate the distance and direction (X and Y change)
                dx, dy = 0, 0
                if thumb_location and prev_thumb_location:
                    dx = thumb_location[0] - prev_thumb_location[0]
                    dy = prev_thumb_location[1] - thumb_location[1]  # Negate dy to flip Y direction
                
                distance =  HandDistanceCalculator()
                # Adjust sensitivity for mouse-like movement
                sensitivity = 10.00 # Adjust sensitivity as needed
                #sensitivity = distance # Adjust sensitivity as needed
                #sensitivity = (distance/100)*sensitivity
                dx *= sensitivity
                dy *= sensitivity
                
                index_location = (int(landmarks.landmark[8].x * frame.shape[1]), int(landmarks.landmark[8].y * frame.shape[0]))
                middle_finger_location = (int(landmarks.landmark[12].x * frame.shape[1]), int(landmarks.landmark[12].y * frame.shape[0]))

                # Detect index finger pinch
                is_index_pinch = index_pinch_detector.detect_index_pinch(thumb_location, index_location)
                
                # Perform a left-click action when index pinch is detected
                #if is_index_pinch:
                    #pyautogui.click(button='left')
            
                # Detect middle finger pinch
                is_middle_finger_pinch = middle_finger_pinch_detector.detect_middle_finger_pinch(thumb_location, middle_finger_location)
                
                #if is_middle_finger_pinch:
                    #pyautogui.click(button='right')
                    
                # Detect triple finger pinch
                is_triple_finger_pinch = triple_finger_pinch_detector.detect_triple_finger_pinch(thumb_location, index_location, middle_finger_location)
                
                if is_triple_finger_pinch:
                    # Scroll in the direction of dx and dy
                    sx=-dx
                    sy=-dy
                    scroll_speed = 3
                    ctypes.windll.user32.mouse_event(0x800, 0, 0, int(sx * scroll_speed), int(sy * scroll_speed))  # Wheel up
                
                else:
                    # Detect index finger pinch
                    is_index_pinch = index_pinch_detector.detect_index_pinch(thumb_location, index_location)
                    
                    # Perform a left-click action when index pinch is detected
                    if is_index_pinch:
                        pyautogui.click(button='left')
                
                    # Detect middle finger pinch
                    is_middle_finger_pinch = middle_finger_pinch_detector.detect_middle_finger_pinch(thumb_location, middle_finger_location)
                    
                    if is_middle_finger_pinch:
                        pyautogui.click(button='right')
            
            
                # Display the coordinates of the thumb tip and sensitivity
                #cv2.putText(frame, f"Sensitivity: {distance_calculator.threshold_multiplier:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(frame, f"Index Pinch: {'Yes' if is_index_pinch else 'No'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_index_pinch else (0, 255, 0), 2)
                #cv2.putText(frame, f"Middle Finger Pinch: {'Yes' if is_middle_finger_pinch else 'No'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_middle_finger_pinch else (0, 255, 0), 2)
                #cv2.putText(frame, f"Triple Finger Pinch: {'Yes' if is_triple_finger_pinch else 'No'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_triple_finger_pinch else (0, 255, 0), 2)

                # Simulate mouse movement based on the calculated distance and direction
                cursor_simulator.move_cursor(dx, dy)

                # Connect the landmarks to form hand lines
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, distance_calculator.mp_hands.HAND_CONNECTIONS)

        # Display the frame with landmarks
        #cv2.imshow("Hand Tracking", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
