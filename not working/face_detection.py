#!/usr/bin/env python3
#
from textwrap import dedent
import cv2
import numpy as np
import math



def detection_faces():
    count = 0
    center_points_frame_prev = []
    
    tracking_objects = {}
    track_id = 0
    ####################FACE DETECTION#############################
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/haarcascade_frontalface_default.xml')

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)
    # To use a video file as input 
    # cap = cv2.VideoCapture('filename.mp4')

    while True:
        # Read the frame
        _, img = cap.read()

        count = count + 1
        
        # Points current frame
        center_points_frame_cur = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
           
           
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            # masked = cv2.bitwise_and(img, img, mask=mask)
            # big_mask = cv2.resize(masked, (0, 0), fx=2, fy=2)
            # #cv2.imshow("Rectangular Mask", big_mask)
            
            
            
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            center_points_frame_cur.append((cx, cy))
        
        if count <= 2:
            for pt in center_points_frame_cur:
                for pt2 in center_points_frame_prev:
                    distance = math.hypot(pt2[0]-pt[0], pt[1]-pt[1])
                    
                    if distance < 5:
                        tracking_objects[track_id] = pt
                        track_id = track_id + 1 
        else:
            for pt in center_points_frame_cur:
                for pt2 in tracking_objects.items():
                    if distance < 5:
                        tracking_objects[object_id] = pt
                        
                    
        for object_id, pt in tracking_objects.items():
            print("here")
            cv2.circle(img, pt, 5, (0,0,255), -1)
            cv2.putText(img, str(object_id), (pt[0],pt[1]-7), 0,1,(0,0,255))
                    
        
            
           
        # Display
        cv2.imshow('img', img)
        
        #make a copy of the points
        center_points_frame_prev = center_points_frame_cur.copy()
        
        # Stop if "Q" key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
#########################################################################
    


if __name__ == "__main__":
    detection_faces()
    
    
    