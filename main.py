import cv2
import pyautogui
from cvzone.HandTrackingModule import HandDetector

# Variables
width, height = 1280, 720
gesture_threshold = 300
pressed_delay = 10
pressed_left, pressed_right = False, False
left_counter, right_counter = 0, 0

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img = cap.read()
    #img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)

    if hands:
        # Kind of an ugly fix, will not work when there are multiple people
        hand = None
        for h in hands:
            if h["type"] == "Right":
                hand = h
                break
        if hand:
            fingers = detector.fingersUp(hand)
            cx, cy = hand["center"]
            # If hand is at face level
            if cy <= gesture_threshold:
                # Gesture 1 - Left
                if fingers[1:] == [1, 0, 0, 0] and not pressed_left:
                    print("Left")
                    pyautogui.press("left")
                    pressed_left = True

                # Gesture 2 - Right
                if fingers[1:] == [0, 0, 0, 1] and not pressed_right:
                    print("Right")
                    pyautogui.press("right")
                    pressed_right = True

    # Button pressed cooldown
    if pressed_left:
        if left_counter < pressed_delay:
            left_counter += 1
        else:
            left_counter = 0
            pressed_left = False

    if pressed_right:
        if right_counter < pressed_delay:
            right_counter += 1
        else:
            right_counter = 0
            pressed_right = False

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
