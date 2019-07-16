
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
import os
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
#external device is used 
cap = cv2.VideoCapture(1)
json_file = open('model_dropout.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('modelfer_dropout50.h5')
print("Loaded model from disk")
target = ['0','A', 'B', 'C','D','del', 'E', 'F', 'G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
print(len(target))


# In[2]:


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x=350;y=60;w=200;h=200;   
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
    gesture_crop = img[y:y + h, x:x + w]
    gesture_crop = cv2.resize(gesture_crop, (200, 200))
    gesture_crop = cv2.cvtColor(gesture_crop, cv2.COLOR_BGR2GRAY)
    gesture_crop = np.asarray(gesture_crop)
    gesture_crop = gesture_crop.reshape(1,gesture_crop.shape[0], gesture_crop.shape[1],1)
    print(np.argmax(loaded_model.predict(gesture_crop)))
    result = target[np.argmax(loaded_model.predict(gesture_crop))]
    cv2.putText(img, result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3,cv2.LINE_AA)

 
    cv2.imshow('img', img)   
    k = cv2.waitKey(30) & 0xff 
    #exit on esc
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[4]:


loaded_model.summary()

