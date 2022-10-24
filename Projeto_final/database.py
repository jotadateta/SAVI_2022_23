import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def database(person_file):
    faces_database = []
    files = glob.glob("faces/*.png")
    for file in files:
        print(file)
        image = cv2.imread(file)
        faces_database.append(image)

    print('Faces DataBase shape:', np.array(faces_database).shape)
    plot_lines = int(len(faces_database)//3+1)
    for i in range(len(faces_database)):
        plt.subplot(plot_lines,3,i+1),plt.imshow(cv2.cvtColor(faces_database[i], cv2.COLOR_BGR2RGB)) #faces_database[i],'gray',vmin=0,vmax=255)
        plt.xticks([]),plt.yticks([])
    plt.draw(); plt.pause(0.01)

    # Este imshow pode dar asneira, mas o que precisa e que o ficheiro person_file de entrada seja a foto da pessoa que o programa reconhece
    #cv2.imshow("Person Recognised", cv2.imread(person_file), cv2.IMREAD_GRAYSCALE)
    
# if __name__ == '__main__':
#     database("/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces/jota_1.png")

def database_show():
    

    #path
    path='/home/jota/Documents/SAVI/savi_22-23/SAVI_Trabalho1/faces'
    images=os.listdir(path)
    #type(images)
    #len(images)
    
    
    for img in images:
        img_arr=cv2.imread(os.path.join(path,img))
        plt.figure
        plt.imshow(img_arr)
