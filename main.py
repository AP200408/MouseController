import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

def get_landmark_position(landmark, frame_shape):
    height, width, _ = frame_shape
    return int(landmark.x * width), int(landmark.y * height)

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Previous positions and states to determine gestures
prev_index_y = 0
prev_thumb_index_distance = 0
click_time = 0
dragging = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move the mouse
            pyautogui.moveTo(x, y)

            # Get the tip of the thumb
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip_position = get_landmark_position(index_finger_tip, frame.shape)
            thumb_tip_position = get_landmark_position(thumb_tip, frame.shape)

            # Get the tip of the middle finger
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_tip_position = get_landmark_position(middle_finger_tip, frame.shape)

            # Distance between index finger tip and thumb tip
            thumb_index_distance = distance(index_tip_position, thumb_tip_position)

            # Distance between index finger tip and middle finger tip for scrolling
            index_middle_distance = distance(index_tip_position, middle_tip_position)

            # Click detection
            if thumb_index_distance < 40:  # Adjust this threshold based on testing
                if time.time() - click_time > 0.5:  # Simple debounce
                    pyautogui.click()
                    click_time = time.time()

            # Scroll detection based on vertical movement of the index finger
            if index_middle_distance < 40:  # Adjust this threshold based on testing
                if y < prev_index_y - 10:  # Scroll up
                    pyautogui.scroll(20)
                elif y > prev_index_y + 10:  # Scroll down
                    pyautogui.scroll(-20)
                prev_index_y = y

            # Drag and drop detection
            if thumb_index_distance < 40 and not dragging:
                pyautogui.mouseDown()
                dragging = True
            elif thumb_index_distance >= 40 and dragging:
                pyautogui.mouseUp()
                dragging = False

            prev_thumb_index_distance = thumb_index_distance

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
