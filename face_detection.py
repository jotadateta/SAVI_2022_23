#!/usr/bin/env python3
#
from textwrap import dedent
import cv2
import numpy as np


def detection_faces():
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

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # a mask is the same size as our image, but has only two pixel
            # values, 0 and 255 -- pixels with a value of 0 (background) are
            # ignored in the original image while mask pixels with a value of
            # 255 (foreground) are allowed to be kept
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            masked = cv2.bitwise_and(img, img, mask=mask)
            big_mask = cv2.resize(masked, (0, 0), fx=2, fy=2)
            cv2.imshow("Rectangular Mask", big_mask)
        
        # Display
        cv2.imshow('img', img)

        # Stop if "Q" key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
#########################################################################
    


if __name__ == "__main__":
    detection_faces()