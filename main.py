import cv2
import mediapipe as mp
import pyautogui
import time

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        
        self.scroll_cooldown = 0.4  # seconds between scroll actions
        self.last_action_time = time.time()
        self.prev_hand_position = None
        
        # Gesture thresholds
        self.scroll_threshold = 0.05  
        
    def detect_gestures(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        
        results = self.hands.process(rgb_frame)
        
        
        gesture = "none"
        
        if results.multi_hand_landmarks:
            
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            
            palm_center = self.calculate_palm_center(hand_landmarks, frame_width, frame_height)
            
            
            if self.prev_hand_position is not None:
                vertical_movement = palm_center[1] - self.prev_hand_position[1]
                
                if time.time() - self.last_action_time > self.scroll_cooldown:
                    if vertical_movement > self.scroll_threshold * frame_height:
                        gesture = "scroll_down"
                        self.last_action_time = time.time()
                    elif vertical_movement < -self.scroll_threshold * frame_height:
                        gesture = "scroll_up"
                        self.last_action_time = time.time()
                
                
                cv2.putText(frame, f"V-Move: {vertical_movement:.2f}", 
                            (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
            
            # Update previous hand position
            self.prev_hand_position = palm_center
            
            
            cv2.circle(frame, (int(palm_center[0]), int(palm_center[1])), 
                       10, (0, 255, 0), -1)
        else:
            self.prev_hand_position = None
        
        
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
        
        return frame, gesture
    
    def calculate_palm_center(self, hand_landmarks, frame_width, frame_height):
        """Calculate the center of the palm based on hand landmarks."""
        
        palm_points = [0, 1, 5, 9, 13, 17] 
        x_sum = y_sum = 0
        
        for point in palm_points:
            x_sum += hand_landmarks.landmark[point].x * frame_width
            y_sum += hand_landmarks.landmark[point].y * frame_height
        
        return [x_sum / len(palm_points), y_sum / len(palm_points)]
    
    def execute_action(self, gesture):
        """Execute the action corresponding to the detected gesture."""
        if gesture == "scroll_down":
            pyautogui.scroll(-100)  
            print("Scrolling down")
        elif gesture == "scroll_up":
            pyautogui.scroll(100)   
            print("Scrolling up")

def main():
    
    controller = GestureController()
    
    
    cap = cv2.VideoCapture(0)
    
    print("=== Hand Gesture Social Media Controller ===")
    print("Hold your hand in front of the camera")
    print("Move your hand up/down to scroll")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        
        frame = cv2.flip(frame, 1)
        
        
        frame, gesture = controller.detect_gestures(frame)
        
        
        controller.execute_action(gesture)
        
        
        cv2.imshow('Hand Gesture Controller', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()