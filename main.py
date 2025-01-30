import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Frames")
ax.set_ylabel("Variance")
ax.set_title("Variance of Hand Landmark Positions")
right_line, = ax.plot([], [], 'r-', label="Right Hand")
left_line, = ax.plot([], [], 'b-', label="Left Hand")
ax.legend()
#plt.show(block=False)

# Data storage
frame_count = 0
max_frames = 100
right_history = deque(maxlen=max_frames)
left_history = deque(maxlen=max_frames)
frames = deque(range(max_frames), maxlen=max_frames)

# Start capturing video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    right_dist, left_dist = 0, 0

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Compute distance between thumb and middle finger
            thumb_pos = np.array([
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            ])
            pinky_pos = np.array([
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            ])
            
            dist = np.linalg.norm(thumb_pos - pinky_pos)

            # Determine left or right hand
            label = handedness.classification[0].label  # "Left" or "Right"
            if label == "Right":
                right_dist = dist
            else:
                left_dist = dist

    # Update data history
    frames.append(frame_count)  # Ensure frames always grows
    right_history.append(right_dist)  
    left_history.append(left_dist)  

    # Ensure all lists are the same length
    while len(right_history) < len(frames):
        right_history.append(0)  # Fill missing values with 0
    while len(left_history) < len(frames):
        left_history.append(0)

    # Update plot
    right_line.set_data(frames, right_history)
    left_line.set_data(frames, left_history)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

    frame_count += 1

    # Show the video feed
    cv2.imshow("Hand Tracking", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.close("all")
