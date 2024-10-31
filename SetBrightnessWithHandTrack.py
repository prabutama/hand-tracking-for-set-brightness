import cv2
import numpy as np
import HandTrackingModule as htm
import math
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(detectionCon=1)
min_distance = 20
max_distance = 250

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)

        brightness = np.interp(length, [min_distance, max_distance], [0, 100])
        sbc.set_brightness(brightness)

        # Visualisasi volume pada layar
        cv2.putText(img, f'Brightness: {int(brightness)} %', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow("Gesture-Based Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
