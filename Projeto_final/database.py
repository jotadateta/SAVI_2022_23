import cv2
from matplotlib import pyplot as plt
import glob
import numpy as np

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
    plt.show()

    # Este imshow pode dar asneira, mas o que precisa e que o ficheiro person_file de entrada seja a foto da pessoa que o programa reconhece
    cv2.imshow("Person Recognised", cv2.imread(person_file), cv2.IMREAD_GRAYSCALE)