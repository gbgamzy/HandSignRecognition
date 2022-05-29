import cv2
import mediapipe as mp
from tensorflow.keras import Input, Model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

ds = pd.read_csv('./dataset/dataset.csv')
# ds = pd.read_csv('dataset_reduced.csv')
# ds = pd.read_csv('german_dataset.csv')
# ds = pd.read_csv('german_2d_dataset.csv')

labels = pd.unique(ds.label)

print("Which model do you want to run?")
print("""
  1. ISL v1.0
  2. ISL v1.1
  3. ISL v2.0
  4. ISL_noise
  5. GSL v1.0
  6. GSL_2d v1.0
""")

i = int(input())

while(True):
  if( i == 1):
    ds = pd.read_csv('./dataset/dataset_correct.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/isl_model.h5')
    break
  if( i == 2):
    ds = pd.read_csv('./dataset/dataset_correct.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/isl_model_1_1.h5')
    break
  if( i == 3):
    ds = pd.read_csv('./dataset/dataset_correct.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/isl_model_2.h5')
    break
  if( i == 4):
    ds = pd.read_csv('./dataset/dataset_reduced.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/isl_model_with_noise.h5')
    break
  if( i == 5):
    ds = pd.read_csv('./dataset/german_dataset.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/gsl_model.h5')
    break
  if( i == 6):
    ds = pd.read_csv('./dataset/german_2d_dataset.csv')
    labels = pd.unique(ds.label)
    model = keras.models.load_model('./models/gsl_model_2d.h5')
    break
  else:
    print("wrong input")



# model = keras.models.load_model('isl_model.h5')
# model = keras.models.load_model('isl_model_1_1.h5')
# model = keras.models.load_model('isl_model_2.h5')
# model = keras.models.load_model('isl_model_with_noise.h5')
# model = keras.models.load_model('gsl_model.h5')
# model = keras.models.load_model('gsl_model_2d.h5')


# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        list = []
        cv2.flip(image, 1)
            

        if(i != 6):
          for i in range(0, 21):
              list.append(hand_landmarks.landmark[i].x)
              list.append(hand_landmarks.landmark[i].y)
              list.append(hand_landmarks.landmark[i].z)
        else:
          for i in range(0, 21):
              list.append(hand_landmarks.landmark[i].x)
              list.append(hand_landmarks.landmark[i].y)
              # list.append(hand_landmarks.landmark[i].z)
              





        list = np.array(list)
        if( i == 6):
          list = np.reshape(list,(1,21,2))
        else:
          list = np.reshape(list,(1,21,3))
        # list = np.reshape(list,(1,63,1))
        
        print(list)
        ans = model.predict(list)
        print(ans.shape)
        index = np.argmax(ans)
        print(labels[index])

        cv2.putText(image, str(labels[index]), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()