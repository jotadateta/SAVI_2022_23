#!/usr/bin/env python3
#
import csv
from copy import deepcopy
from turtle import color

import cv2
import numpy as np
from colorama import Fore, Style, Back


class Tracker():

    def __init__(self, detection, id, image, tracker):
        self.detection = detection
        self.id = id
        self.tracker = tracker
        
        
        self.track_update(image)


    def track_update(self, img):

        sucess, bbox = self.tracker.update(img)
        print("estado = " + str(sucess) + " bbo = " + str(bbox))
        self.drawBox(img, bbox)
    
    def drawBox(self, img, bbox):
        x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,140,0),3)

        

