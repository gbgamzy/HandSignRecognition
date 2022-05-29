import cv2
import mediapipe as mp
import csv
from csv import writer
import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def writeListToCsv(list_data):
    
    
    with open('./dataset_reduced.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list_data)
        f_object.close()


# For static images:
IMAGE_FILES = []
images = glob("//Users//gautambhateja//Desktop//Projects//Minor project//IndianSignLanguage//*//*.jpg")

for image in images:
    
    
    IMAGE_FILES.append(image)



def processImage(image):
    
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        # for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
        head, tail = os.path.split(file)
        # print(head)
        
        x = head.split('/')
        # print(x)
        # print(tail)
        
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # results = hands.process(image)

        # Print handedness and draw hand landmarks on the image.
        
        if not results.multi_hand_landmarks:
            return
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            list = []
            list.append(str(head[-1]))

        #    print("------")
            for i in range(0, 21):
                list.append(hand_landmarks.landmark[i].x)
                list.append(hand_landmarks.landmark[i].y)
                list.append(hand_landmarks.landmark[i].z)
                # print('hand_landmarks:', hand_landmarks)
        writeListToCsv(list)

print(len(IMAGE_FILES))
a = 1
for idx, file in enumerate(IMAGE_FILES):
    print(a)
    a += 1
    image = Image.open(file)
    image1 = image.rotate(20)
    image2 = image.rotate(340)

    image = np.array(image)
    image1 = np.array(image1)
    image2 = np.array(image2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    processImage(image)
    processImage(image1)
    processImage(image2)


print('done')