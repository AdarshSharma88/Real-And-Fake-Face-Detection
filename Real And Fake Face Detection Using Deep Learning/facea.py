import cv2
#import face_recognition as fr

import face_recognition as fr
import numpy as np
import os
vid=cv2.VideoCapture(0)
from keras.models import load_model

model=load_model(r'C:\Users\adars\OneDrive\Documents\Visual Studio 2022\Python\Real And Fake Face Detection Using Deep Learning\model.h5');

process_this_frame=True
status="Unknown"
face_encodings = []
face_names = []
while(True):
    ret,frame=vid.read()
    
    xframe=cv2.resize(frame,(224,224))

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    
    frame=cv2.flip(frame,1)
    if process_this_frame:
        small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rbg_small_frame=small_frame[:,:,::-1]
        face_locations=fr.face_locations(rbg_small_frame)

    process_this_frame=not process_this_frame
    print(face_locations)
    for(top,right,bottom,left) in (face_locations):
        top*=4
        right*=4
        bottom*=4
        left*=4
        
        cv2.rectangle(frame,(left-30,top-60),(right+30,bottom+30),(5,225,0),2)
        r = max(bottom, left) / 2
        centerx = top + bottom / 2
        centery = right + left / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        pred=model.predict(np.expand_dims(xframe,axis=0))
        if np.argmax(pred)==1:
            status='Real'
        else:
            status='Fake'
        
        #cv2.rectangle(frame, (left-30, bottom - 5), (right+30, bottom+30), (5,225,0), cv2.FILLED)
        #cv2.putText(frame,name,(left+6,bottom+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1,cv2.FILLED)
    
        cv2.putText(frame, status, org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
vid.release()
cv2.destroyAllWindows()