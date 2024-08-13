import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import POINTER
from comtypes import GUID

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}"),  # IAudioEndpointVolume GUID
    CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Set up video capture
cap = cv2.VideoCapture(0)

def adjust_volume(hand_landmarks):
    if hand_landmarks:
        # Use the distance between index and thumb fingers to adjust volume
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # Calculate the distance between index finger tip and thumb tip
        distance = np.sqrt(
            (index_tip.x - thumb_tip.x) ** 2 +
            (index_tip.y - thumb_tip.y) ** 2
        )

        # Map the distance to volume range (0.0 to 1.0)
        volume_level = max(0.0, min(1.0, distance))
        volume.SetMasterVolumeLevelScalar(volume_level, None)
        print(f"Volume Level: {volume_level:.2f}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            adjust_volume(hand_landmarks)

    # Show the frame
    cv2.imshow('Hand Volume Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
