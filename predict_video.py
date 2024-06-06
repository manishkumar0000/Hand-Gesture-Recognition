from tensorflow import keras 
import keras_preprocessing
# from keras_applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from keras import models
from tensorflow.keras.models import load_model
from collections import deque
import mediapipe as mp
import uuid
import os
import numpy as np
import pickle
import cv2

model = load_model("/home/prachetas/hagrid/ges2.h5")

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
_, frame = cap.read()



h, w, c = frame.shape

while True:
    _, frame = cap.read()

    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                start_row, start_col = int(y_min-25), int(x_min-25)
                end_row, end_col = int(y_max+25), int(x_max+25)
                cropped = frame[start_row:end_row , start_col:end_col]
                cropped=cv2.flip(cropped,1)
                cv2.imshow("Cropped" , cropped)

                frame = cv2.resize(cropped, (81, 81))

                img_data = np.expand_dims(frame, axis=0)
                img_data = preprocess_input(img_data)
                #preds = model_resnet.predict(img_data)
                preds = model.predict(img_data)

                pred_class=(list(preds[0])).index(max((list(preds[0]))))
                

                labels=['no_gesture','call','dislike','fist','four','like','mute','ok','one','palm','peace','peace_inverted','rock','stop','stop_inverted','three','three2','two_up','two_up_inverted']



                text = "Gesture: {}".format(labels[pred_class])
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
                cv2.imshow("Output", output)
    

    # pass croppedimage

    

    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


