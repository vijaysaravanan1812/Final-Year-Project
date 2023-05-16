import cv2
import os
import glob
import numpy as np
import shutil
from random import randint

import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from skimage.transform import resize
from keras import models 
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, LeakyReLU, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from keras.layers import Reshape
from keras import layers
import datetime
from tensorflow.keras.optimizers import Adam

from keras import initializers


def predict_face(path):

    with open('dcgan.json', 'r') as json_file:
        json_savedModel= json_file.read()
        generator = tf.keras.models.model_from_json(json_savedModel)
        generator.load_weights('500_wg_04-07-20_47.h5')
        generator.compile(loss="mean_squared_error", optimizer = Adam(lr=0.00002))
    print("Model compiled")

    img = mpimg.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img.shape
    plt.imshow(img)
    img = resize(img, (64 , 64), anti_aliasing=False)

    img = np.expand_dims(img, axis=-1)
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    
    
    pred = generator.predict(img)

    # plt.imshow(np.reshape(pred, (64,64)), cmap = "gray")
    pred = np.reshape(pred, (64,64))

    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    print("pred shape " , pred.shape)

    low_res_img = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # low_res_img = resize(img, (32 , 32), anti_aliasing=False)
    low_res_img = Image.fromarray(low_res_img)
    low_res_img.save('static/Predicted_face/low_resolution_image.png')
    
    print("in_predict")
    with open('model 64-256.json', 'r') as json_file:
        json_savedModel= json_file.read()
        generator = tf.keras.models.model_from_json(json_savedModel)
        generator.load_weights('model.h5')
        generator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    print("Model compiled")
    
        
    # Do something with the file
    low_resolution_image = cv2.imread('static/Predicted_face/low_resolution_image.png')
    # Compare the shape of the image with a constant value
    if low_resolution_image.shape >= (64, 64, 3):
        low_resolution_image = cv2.resize(low_resolution_image, (64 , 64))
    else:
        print("error")
    #Change images from BGR to RGB for plotting. 
    #Remember that we used cv2 to load images which loads as BGR.
    low_resolution_image = cv2.cvtColor(low_resolution_image, cv2.COLOR_BGR2RGB)
    low_resolution_image = low_resolution_image / 255.
    low_resolution_image = np.expand_dims(low_resolution_image, axis=0)
    high_resolution_image = generator.predict(low_resolution_image)

    high_resolution_image = np.squeeze(high_resolution_image, axis=0)
    high_resolution_image = cv2.normalize(high_resolution_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = Image.fromarray(high_resolution_image)

    # save the image file
    img.save('static/High_resolution_face/high_resolution_image.png')
    return 'static/High_resolution_face/high_resolution_image.png'

