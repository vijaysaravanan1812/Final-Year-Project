import cv2
import os

import glob
import cv2
#globbing utility.
import glob
import os
import numpy as np
#select the path
#I have provided my path from my local computer, please change it accordingly

# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.utils import img_to_array

from tensorflow.keras.utils import load_img
from numpy import savez_compressed
import random

# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam

from matplotlib import pyplot
import tensorflow as tf
from numpy import load
from matplotlib import pyplot
from PIL import Image
import tensorflow as tf
import time


def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = tf.keras.preprocessing.image.img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = tf.keras.backend.expand_dims(pixels, 0)
	return pixels

def is_grayscale(image_path):
    image = Image.open(image_path).convert('RGB')
    r, g, b = image.split()
    if r == g == b:
        return True
    else:
        return False


def predict(path):
    img = load_image(path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    model = tf.keras.models.load_model('model_016560.h5', compile = False)
    gen_img = model.predict(img , steps = 1)
    gen_img = np.squeeze(gen_img)
    gen_img = cv2.normalize(gen_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gen_img = Image.fromarray(gen_img)
    # Generate the high-resolution image and record the inference time

    gen_img.save('static/front_view_face_img/gen_img.png')
    return 'static/front_view_face_img/gen_img.png'

