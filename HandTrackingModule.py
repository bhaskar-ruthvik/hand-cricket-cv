import cv2 as cv
import mediapipe as mp

class HandTracker:
    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self,image,draw=True):
        imgRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        h,w,c = image.shape
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw and handLms.landmark[0].x*w>150 and handLms.landmark[0].x*w<510 and handLms.landmark[0].y*h>150 and handLms.landmark[0].y*h<330:
                    self.mpDraw.draw_landmarks(image,handLms,self.mpHands.HAND_CONNECTIONS)
        return image
    
    def findPosition(self,image,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append((id,cx,cy))
                if draw:
                    cv.circle(image,(cx,cy),7,(255,0,255),-1)
        return lmList
    