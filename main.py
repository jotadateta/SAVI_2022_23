from ssl import OPENSSL_VERSION
import string
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import glob
import os, sys
from functions import Detection, Tracker
import face_recognition
import math


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


def database(all_persons_files,persons_in_frame):
    counter = 1
    #faces_database = []
    #files = glob.glob("faces/*.png")
    #for file in files:        
    #    image = cv2.imread(file)
    #    faces_database.append(image)

    plot_lines = int((len(all_persons_files)-1)//3+1)
    for i in range(len(all_persons_files)):
        plt.subplot(plot_lines,3,i+1),plt.imshow(cv2.cvtColor(all_persons_files[i], cv2.COLOR_BGR2RGB)) #faces_database[i],'gray',vmin=0,vmax=255)
        plt.xticks([]),plt.yticks([])
    plt.draw()
    plt.pause(0.0001)

    # Este imshow pode dar asneira, mas o que precisa e que o ficheiro person_file de entrada seja a foto da pessoa que o programa reconhece
    for person_in_frame in persons_in_frame:
        window_name = "Person Recognised "+str(counter)
        cv2.imshow(window_name, person_in_frame)
        counter+=1





def main():
    # ---------------------
    # Define photos path
    # ---------------------
    
    path = "C:\\Users\\Luis Pires\\Documents\\SAVI\\SAVI_Trabalho1\\SAVI_2022_23\\faces"


    known_face_names = []
    known_face_encodings = []
    faces_known = []
    # -----------------------------------------
    # for loop to read all photos inside folder
    # -----------------------------------------
    for image in os.listdir(path):
        face_image = face_recognition.load_image_file(f"{path}/{image}")
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
    
    
    # -------------
    # Read weabcam
    # -------------
    cap = cv2.VideoCapture(0)

    # -----------
    # Processing
    # -----------
    while True:
        # -------------
        # Get the frame
        # -------------
        _, frame_rgb = cap.read()
        frame_counter +=1

        persons_files = []
        
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
                        name = known_face_names[best_match_index]
                        persons_files.append(faces_known[best_match_index])
                        confidence = face_confidence(face_distances[best_match_index])

                    if name == "Unknown":
                        person_name = str(input("Person not recognised, please insert name to save in database: "))
                        person_name = person_name.replace("\W","_")
                        print(person_name)
                        known_face_names.append(person_name)
                        known_face_encodings.append(face_encoding_processed)
                        faces_known.append(frame)

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
            print(last_detection_id)
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

        cv2.imshow('window_name',image_gui) # show the image
        database(faces_known,persons_files)
        

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