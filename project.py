# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:02:20 2022

@author: sulta
"""

# Import libraries
import cv2
import numpy as np
import vtrack_svm
from skimage.feature import hog
from PIL import Image

svm = vtrack_svm.load_svm_classifier('svm_cars2.pickle')

#Web Camera
#cap = cv2.VideoCapture('20221103_122449_4.mp4')
#cap = cv2.VideoCapture('city_traffic_01.mp4')
cap = cv2.VideoCapture('VID-20221116-WA0014.mp4')

cap.set(3, 1280)
cap.set(4, 720)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

min_width_rectangle = 20
min_height_rectangle = 20
count_line_positionx = 150 #X start
count_line_positionxE = abs(width//2)-60 # X end
decount_line_position = height - 170 #level y position
count_line_position = decount_line_position-90 #count y position

# Initialize Substructor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

def is_car(img):
    img = np.asarray(img)
    img = cv2.cvtColor(cv2.resize(img, (96,64)), cv2.COLOR_RGB2GRAY)
    hog_features = hog(img,
                       pixels_per_cell=(16,16),
                       cells_per_block=(2,2))
    y_pred = svm.predict(np.asarray([hog_features]))
    return bool(int(y_pred))

detect = []
offset = 2  #Alowable error b/w pixel
counter = 0
leave =0

while True:
    ret, video = cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    

# Applying on each frame
    vid_sub = algo.apply(blur)
    dilat = cv2.dilate(vid_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countersahpe, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(video, (count_line_positionx,count_line_position),(count_line_positionxE,count_line_position),(0,127,255), 2)

    cv2.line(video, (count_line_positionx,decount_line_position),(count_line_positionxE,decount_line_position),(0,127,255), 2)

    if len(countersahpe) > 15: continue
    for (i, c) in enumerate(countersahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rectangle) and (h>= min_height_rectangle)
        if not val_counter:
            continue
        #cv2.imshow('test', video[y:y+h, x:x+w])
        obj = video[y:y+h, x:x+w]
        pred = is_car(obj)
        if pred:
            cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(video,"Vehicle No: " + str(counter), (x,y-20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,244,0),2)


        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(video, center, 4, (0,0,255), -1)

        #cunt detect
        for (x,y) in detect:
            if y<(count_line_position + offset) and  y>(count_line_position - offset):
                if x<(count_line_positionxE):
                    counter+=1
                    cv2.line(video, (count_line_positionx,count_line_position),(count_line_positionxE,count_line_position),(0,127,255), 2)
                    detect.remove((x,y))
                    print("Vehicle No: "+ str(counter))
                  
        #leave detect
        for (x,y) in detect:
            if y<(decount_line_position + offset) and  y>(decount_line_position - offset):
                if x<(count_line_positionxE):
                    leave+=1
                    cv2.line(video, (count_line_positionx,decount_line_position),(count_line_positionxE,decount_line_position),(0,127,255), 2)
                    detect.remove((x,y))
                    print("leave No: "+ str(leave))
    #the big red words in the screan
    cv2.putText(video,"Vehicle No: " + str(counter), (0,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),3)
    cv2.putText(video,"leave No: " + str(leave), (0,200),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),3)

    cv2.imshow('Detector',video)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
