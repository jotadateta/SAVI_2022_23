import face_recognition
import os, sys
import cv2
import numpy as np
import math
from tracking import Tracker

#tracker = cv2.TrackerCSRT_create()
#tracker_type = cv2.TrackerCSRT_create()

#tracker = cv2.MultiTracker_create()
tracker = cv2.legacy.MultiTracker_create()


# Helper #make de %of confidence is the person based on photos
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def track_update(img, tracker_id):
        sucess, bbox = tracker.update(img)
        #print("estado = " + str(sucess) + " bbo = " + str(bbox))
        if sucess:
            drawBox(img, bbox, tracker_id)

def drawBox(img, bbox, tracker_id):
    print(bbox)
    for (x, y, w, h)in bbox:
        #x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        
        cv2.rectangle(img,(int(x),int(y)),((int(x)+int(w)),(int(y)+int(h))),(255,140,0),3)
        #cv2.putText(img, str(tracker_id), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    detections = []
    tracker_id = 0
    counted_id = 0
    trackers = []
    bbox_prev = []
    
    
    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        
        # ---------------------------------
        # detects if we have video sources
        # ---------------------------------
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()
            self.detections = []

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # ------------------------------------------
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                # ------------------------------------------
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # --------------------------------------------
                # Find all the faces and face encodings in the current frame of video
                # --------------------------------------------
                self.face_locations = face_recognition.face_locations(rgb_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # ------------------------------------------------
                    # See if the face is a match for the known face(s)
                    # ------------------------------------------------
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'
                    # ----------------------------------------
                    # Calculate the shortest distance to face
                    # ----------------------------------------
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame
            # ---------------------
            # Display the results (BBox)
            # ----------------------
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # -------------------------------
                # Create the frame with the name
                # -------------------------------
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                w = right-left
                h = bottom-top
                bbox = left,top,w,h
                # ----------------------------
                # Create a list of detections
                # ----------------------------
                self.detections.append(bbox)
                #print(self.detections)      
                size = int(len(self.detections)) #number of persons
                #print("counted id = " + str(self.counted_id))
                #print(bbox)
                
                if bbox != None:
                    #print(bbox)
                    # --------------------------------------------
                    #Initalize the tracker for each person founded
                    # --------------------------------------------
                    #trackers.init (frame,bbox) 
                    #tracker_spec = OPENCV_OBJECT_TRACKERS[cv2.TrackerCSRT_create]()
                    tracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)

                    #print("pessoas = " + str(size))
                    #print("tracker id = " + str(self.tracker_id))
                    self.tracker_id += 1
                    
                
            
            # ----------------------------
            # Updte to tracker inside loop
            # ----------------------------
            
            track_update(frame, self.tracker_id)
            
                        
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
