#!/usr/bin/env python3


from typing import Counter
import cv2
import numpy as np
from copy import copy, deepcopy
import os, sys
import os, shutil
from functions import Detection, Tracker
import face_recognition
import math
from database import database_show
import tkinter as tk
import pyttsx3
import glob
import matplotlib.pyplot as plt


def database(person_file, persons_in_frame):
    Counter = 1
    print('Faces DataBase shape:', np.array(person_file).shape)
    plot_lines = int(len(person_file)//3+1)
    for i in range(len(person_file)):
        plt.subplot(plot_lines,3,i+1),plt.imshow(cv2.cvtColor(person_file[i], cv2.COLOR_BGR2RGB)) #faces_database[i],'gray',vmin=0,vmax=255)
        plt.xticks([]),plt.yticks([])
    plt.draw(); plt.pause(0.01)

    for person_in_frame in persons_in_frame:
        window_name = "person recognized" + str(Counter)
        cv2.imshow(window_name,person_in_frame)
        Counter += 1
    # Este imshow pode dar asneira, mas o que precisa e que o ficheiro person_file de entrada seja a foto da pessoa que o programa reconhece
    #cv2.imshow("Person Recognised", cv2.imread(person_file), cv2.IMREAD_GRAYSCALE)
    
# if __name__ == '__main__':
#     database("/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces/jota_1.png")



# ------------------------------------------------------------
# Helper #make de %of confidence is the person based on photos
# ------------------------------------------------------------
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# ---------------------
# clean database folder
# ---------------------
def cleanfolder():
    folder = '/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))





def main():
    # -----------
    # clean database folder
    # -----------
    #cleanfolder()
    
    
    
    # ---------------------
    # Define photos path
    # ---------------------
    
    path = "/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces"
    
    known_face_names = []
    known_face_encodings = []
    
    

    faces_known = []
    # -----------------------------------------
    # for loop to read all photos inside folder
    # -----------------------------------------
    for image in os.listdir(path):
        face_image = face_recognition.load_image_file(f"/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces/{image}")
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_names.append(image)
        known_face_encodings.append(face_encoding)
        faces_known.append(face_image)
        
    # ----------
    # variables
    # ----------
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8
    frame_counter = 0
    face_names = []
    Person_previous_frame = []
    
    
    # -------------
    # Read weabcam
    # -------------
    cap = cv2.VideoCapture(0)

    # -----------
    # Processing
    # -----------
    while True:
        
        
        Person_current_frame = []
        Person_files = []
        
        # ----------
        # Database
        # ---------
        # for img in os.listdir(path):
        #     img_arr=cv2.imread(os.path.join(path,img))
        #     plt.figure
        #     plt.imshow(img_arr)
        
            
        
        # -------------
        # Get the frame
        # -------------
        _, frame_rgb = cap.read()
        frame_counter +=1
        
        # -----------------------------
        # mirror the web for processing
        # -----------------------------
        frame = cv2.flip(frame_rgb, 1)
        
        # --------------------------------------------
        # Convert image into gray and copy for drawing
        # --------------------------------------------
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_gui = deepcopy(frame)



        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        # ------------------------------------------
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # ------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # --------------------------------------------
        # Find all the faces and face encodings in the current frame of video
        # --------------------------------------------
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        unknown_person = False
        face_names = []
        for face_encoding_processed in face_encodings:
                    # ------------------------------------------------
                    # See if the face is a match for the known face(s)
                    # ------------------------------------------------
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_processed)
                    name = "Unknown"
                    confidence = '???'
                    # ----------------------------------------
                    # Calculate the shortest distance to face
                    # ----------------------------------------
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_processed)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        confidence = face_confidence(face_distances[best_match_index])
                        if confidence > str(50):
                            name = known_face_names[best_match_index]
                            Person_current_frame.append(name)
                            #Person_files.append(faces_known[best_match_index])
                            
                        else:
                            person = input("Quem estou a ver?")
                            name = str(person)
                            known_face_names.append(name)
                            known_face_encodings.append(face_encoding_processed)
                            unknown_person = True
                            
                            
                    else:
                        person = input("Quem estou a ver?")
                        name = str(person)
                        print(name)
                        
                        known_face_names.append(name)
                        known_face_encodings.append(face_encoding_processed)
                        unknown_person = True
                    face_names.append(f'{name} ({confidence})')
        # ------------------------------------------
        # Create Detections bbox
        # ------------------------------------------
        detections = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            w = right-left
            h = bottom-top
            x1 = left
            y1 = top
            
            detection = Detection(x1, y1, w, h, image_gray, id=detection_counter, stamp=stamp, person_name = name)
            detection_counter += 1
            detection.draw(image_gui, name)
            detections.append(detection)
            
            if unknown_person:
                
                # Cropping an image
                cropped_image = image_gui[top:bottom, left:right]

                # Display cropped image
                cv2.imshow("cropped", cropped_image)
                
                # Save cropped image
                cv2.imwrite("/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces/" + name + ".png", cropped_image)
                
                #faces_known.append[cropped_image]
                
        if len(Person_previous_frame) < len(Person_current_frame):
            engine = pyttsx3.init()
            engine.say("Hello")
            engine.runAndWait()
            
        # ------------------------------------------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                if tracker.active:
                    tracker_bbox = tracker.detections[-1]
                    iou = detection.computeIOU(tracker_bbox)
                    # print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                    if iou > iou_threshold: # associate detection with tracker 
                        tracker.addDetection(detection, image_gray)

        # ------------------------------------------
        # Track using template matching
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            #print(last_detection_id)
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                print('Tracker ' + str(tracker.id) + ' Doing some tracking')
                tracker.track(image_gray)

        # ------------------------------------------
        # Deactivate Tracker if no detection for more than T
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            tracker.updateTime(stamp)

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        for detection in detections:
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, id=tracker_counter, image=image_gray)
                tracker_counter += 1
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------

        # Draw trackers
        for tracker in trackers:
            if tracker.active:
                tracker.draw(image_gui)


            # win_name= 'T' + str(tracker.id) + ' template'
            # cv2.imshow(win_name, tracker.template)

        # for tracker in trackers:
            # print(tracker)
            
        Person_previous_frame = deepcopy(Person_current_frame)

        cv2.imshow('window_name',image_gui) # show the image
        database(faces_known,Person_files)
        
        
        if cv2.waitKey(50) == ord('q'):
            break

        frame_counter += 1


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
            

# -----------------------
# run main function
# -----------------------

if __name__ == '__main__':
  main()
    



