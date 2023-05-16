from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow
import cv2
import os
from PIL import Image

import numpy as np
import json
import video_processing as vp
import Predict_Face as pf
import turn_face
# from transformers import pipeline
# from statistics import mode

app = Flask(__name__) 

# classifier = pipeline("zero-shot-classification")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/get_face')
def get_face():
    return render_template('get_face.html')

@app.route('/turn_face')
def turn_face_angle():
    return render_template('turn_face.html')


@app.route('/get_face_image', methods=['POST'])
def get_face_image():
    try:
        if 'image-upload' not in request.files:
            print('No video file provided')
        file = request.files['image-upload']
        # do something with the video file
        filename = file.filename
        print(filename)
        file.save(os.path.join('static/masked_face/', filename))
        file_path = os.path.join('static/masked_face/', filename )
        print(file_path)
        pf.predict_face(file_path)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # low_res_img = resize(img, (32 , 32), anti_aliasing=False)
        img = Image.fromarray(img)
        img.save('static/masked_face/mask_image.png')
 
        file_path = 'static/masked_face/mask_image.png'
        file_path1 = "static/Predicted_face/low_resolution_image.png" 
        file_path2 = 'static/High_resolution_face/high_resolution_image.png'
        return render_template('get_face.html',high_img= file_path2 ,low_img = file_path1, masked_img = file_path )
        
    except ValueError as e:
        print('error : ' +  str(e))
        return render_template('get_face.html')
    except Exception as e:
        print('error : Internal server error')
        return render_template('get_face.html')
    
    


@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the number from the form data
    
    try:
        if 'video_input' not in request.files:
            print('No video file provided')
        start_frame = int(request.form['start_frame'])
        end_frame = int(request.form['end_frame'])
        video_file = request.files['video_input']
        video_name = video_file.filename
        video_file.save(os.path.join('static/video/', video_name))
        file_path = os.path.join('static/video/', video_name )    
        print(file_path)
        vp.index_frame(file_path , start_frame , end_frame)

    except ValueError as e:
        print('error : ' +  str(e))
    except Exception as e:
        print('error : Internal server error')
    # Process the video and the number here
    # ...
    return render_template('video.html')

def predict(file_path):
    print("in_predict")
    with open('model.json', 'r') as json_file:
        json_savedModel= json_file.read()
        generator = tensorflow.keras.models.model_from_json(json_savedModel)
        generator.load_weights('model.h5')
        generator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    print("Model compiled")
    
        
    # Do something with the file
    low_resolution_image = cv2.imread(file_path)
    # Compare the shape of the image with a constant value
    if low_resolution_image.shape >= (32, 32, 3):
        low_resolution_image = cv2.resize(low_resolution_image, (32 , 32))
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
    img.save('static/results/high_resolution_image.png')
    return 'static/results/high_resolution_image.png'


@app.route('/button-click', methods=['POST'])
def upload():
    try:
        if 'image-upload' not in request.files:
            print('No video file provided')
        file = request.files['image-upload']
    
        filename = file.filename
        print(filename)
        file.save(os.path.join('static/data/', filename))
        file_path = os.path.join('static/data/', filename )
        print(file_path)
        file_path2 = predict(file_path)
        file_path1 = "static/data/" + filename

        return render_template('index.html',result= file_path2 ,image_path = file_path1 )
        
    except ValueError as e:
        print('error : ' +  str(e))
        return render_template('index.html')
    except Exception as e:
        print('error : Internal server error')
        return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        video_name = video_file.filename
        video_file.save(video_name)
        return "Video uploaded successfully!"
    return render_template('video.html')

@app.route('/button-click-to-turn-face', methods=['POST'])
def upload_side_face_img():
    try:
        if 'image-upload' not in request.files:
            print('No video file provided')
        file = request.files['image-upload']
    
        filename = file.filename
        print(filename)
        file.save(os.path.join('static/side_view_face_img/', filename))
        file_path = os.path.join('static/side_view_face_img/', filename )
        print(file_path)
        file_path2 = turn_face.predict(file_path)
        # file_path1 = "static/data/" + filename
        print(file_path2)
        return render_template('turn_face.html',turn_face = file_path2 ,actual_face = file_path )
        
    except ValueError as e:
        print('error : ' +  str(e))
        return render_template('turn_face.html')
    except Exception as e:
        print('error : Internal server error')
        return render_template('turn_face.html')
    
if __name__ == '__main__':
    app.run(debug=True)
