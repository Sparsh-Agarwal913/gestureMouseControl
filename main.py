# all imports
import math
import autopy
import cv2
import mediapipe as mp
import time
import numpy as np


# func returns if a finger is up or down
def fingerup(n, lmlist):
    if lmlist[n][2] < lmlist[n - 2][2]:
        return True
    else:
        return False


# function return distance between two landmarks
def finddis(n, m, lmlist):
    distancebtw = math.sqrt((pow((lmlist[n][1] - lmlist[m][1]), 2) + pow((lmlist[n][1] - lmlist[m][1]), 2)))
    return distancebtw


# function is used to get the resolution of the webcam
def getresolution(cam):
    _, res = cam.read()
    resolution = res.shape[:2]
    return resolution


# variables declaration and assign
smoothing = 55
clocx, clocy = 0, 0
plocx, plocy = 0, 0
frame = 50
cooldown = 0.5
previous_click = 0

# setting up mediapipe
mpHands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1, model_complexity=0)
listlm = []

# setting cv2 and webcam
ws, hs = autopy.screen.size()
cap = cv2.VideoCapture(0)
hcam, wcam = getresolution(cap)
cap.set(3, wcam)
cap.set(4, hcam)
fx = wcam - frame
fy = hcam - frame

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    cv2.rectangle(img, (frame, frame), (fx, fy), (255, 255, 255), 5)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for Id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                listlm.append([Id, cx, cy])
                if len(listlm) >= 20:
                    listlm[Id][1] = cx
                    listlm[Id][2] = cy
                    # print(listlm[8][1],listlm[12][1])
                    x1 = listlm[8][1]
                    x1 = np.interp(x1, (frame, wcam - frame), (0, wcam))
                    y1 = listlm[8][2]
                    y1 = np.interp(y1, (frame, hcam - frame), (0, hcam))
                    if fingerup(8, listlm) and fingerup(12, listlm):
                        print("both \n")
                        distance = finddis(8, 12, listlm)
                        cv2.line(img, (listlm[8][1], listlm[8][2]), (listlm[12][1], listlm[12][2]), (0, 255, 0), 2)
                        print(distance)
                        if distance < 12:
                            time_elapsed = time.time() - previous_click
                            if time_elapsed > cooldown:
                                autopy.mouse.click()
                                previous_click = time.time()
                    elif fingerup(8, listlm) and not fingerup(12, listlm):
                        print("one \n")
                        # move the cursor
                        x2 = np.interp(x1, (0, wcam), (0, ws))
                        y2 = np.interp(y1, (0, hcam), (0, hs))

                        clocx = plocx + (x2 - plocx) / smoothing
                        clocy = plocy + (y2 - plocy) / smoothing

                        print(x1, y1)
                        print(x2, y2)
                        autopy.mouse.move(ws - clocx, clocy)
                        plocx = clocx
                        plocy = clocy
            mpdraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
