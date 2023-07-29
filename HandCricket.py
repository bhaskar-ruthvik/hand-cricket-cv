import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule as htm
import random 

pTime = 0
wCam,hCam = 640,480
cap = cv.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.HandTracker()
indices = [8,12,16,20]
score = 0
while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,0,False)
    if len(lmList)!=0:
        fingers=[]
        thumbOpen = False
        if lmList[4][1]>lmList[3][1]:
            thumbOpen = True
            fingers.append(1)
        for index in indices:
            if lmList[index][2]<lmList[index-2][2]:
                fingers.append(1)
        num = random.randrange(1,7)   
        if num == len(fingers):
            score = 0
            cv.putText(img,'Sorry, you are out!',(150,220),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),4)
            # cv.waitKey(2000)
            # break
        else:
            if len(fingers) ==1 and thumbOpen:
                score+=6
            else: 
                score+=len(fingers)
    cv.rectangle(img,(150,150),(510,330),(255,0,255),5)
    cv.putText(img,f"Score: {int(score)}",(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255))
    cv.imshow("Image", img)
    cv.waitKey(0)