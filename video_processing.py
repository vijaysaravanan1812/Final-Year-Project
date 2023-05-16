import cv2
import os
import glob
import numpy as np
import shutil
from random import randint
import sys
import traceback
import mediapipe as mp
import matplotlib.pyplot as plt


img_number = 1  #Start an iterator for image number.

def mediapipe():
    print("--media pipe--")
    Extracted_face_path = 'C:/Users/VIJAY/FYP_project/webpage 1/static/Face_img'
    Extracted_path = 'C:/Users/VIJAY/FYP_project/webpage 1/static/Extracted'
    if(os.path.isdir(Extracted_face_path)):
        shutil.rmtree(Extracted_face_path )
        print("Deleted Results dir")
        os.makedirs(Extracted_face_path)
        print("Created")
    else:
        os.makedirs(Extracted_face_path)
        print("created new")
    
    print("=====================================================================================")
    print("Filtering image having faces")
    folder= Extracted_path

    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.80)
    i=0
    # print(Extracted_path)
    for filename in os.listdir(folder):
        # print(folder + '/'+ filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:    
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(imgRGB)
        
            print(results)
        if results : 
            if results.detections:
                print('-')
                print(results.detections)
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    # print("kal0",bboxC)
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    print(bbox)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    
                    a = bbox[0]
                    b = bbox[1]

                    c = bbox[2]
                    d = bbox[3]
                    face1=img[b:b+d, a:a+c]
                    # plt.imshow(face1)
                    if face1.shape is not None:
                        
                        path= 'C:/Users/VIJAY/FYP_project/webpage 1/static/Face_img/Extracted' +str(i)+'.jpg'
                        print("Extracted :" + path )
        #                 plt.imshow(img)
        #                 plt.show()
                        cv2.imwrite(path, face1) 
                        i=i+1


def detect_faces(Extracted_path, image_path,display=True):
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    CASCADE="C:/Users/VIJAY/FYP_project/webpage 1/static/File/haarcascade_frontalface_default.xml"
    FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

    
    for x,y,w,h in faces:
        sub_img=image[y-10:y+h+10,x-10:x+w+10]
        os.chdir(Extracted_path) 
        # print("changed directory" + os.getcwd())
        cv2.imwrite(str(randint(0,300))+".jpg",sub_img)
        os.chdir("../")
        cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

    if display:
        cv2.imshow("Faces Found",image)
            # if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
            # 	cv2.destroyAllWindows()

 
def haarcascade(path):
    print("--haarcascade--")
    Extracted_path = 'C:/Users/VIJAY/FYP_project/webpage 1/static/Extracted'
    if(os.path.isdir(Extracted_path)):
        shutil.rmtree(Extracted_path )
        print("Deleted Results dir")
        os.makedirs(Extracted_path)
        print("Created")
    else:
        os.makedirs(Extracted_path)
        print("created new")

    print("=====================================================================================")
    print("Detecting the faces from the frames")
    # if __name__ == "__main__":
    img_list = glob.glob(path)
    for file in img_list:
        if os.path.isdir(file):
            for image in os.listdir(file):
                try:
                    print ("Processing.....",os.path.abspath(os.path.join(file,image)))
                    detect_faces(Extracted_path , os.path.abspath(os.path.join(file,image)),False)
                except:
                    print ("Could not process ",os.path.abspath(os.path.join(file,image)))
        else:
            detect_faces(Extracted_path, file)
    # try:
    mediapipe()    
    # except ValueError as e:
    #     print('error : ' +  str(e))
    # except Exception as e:
    #     print('error : Internal server error')

img_number = 1  #Start an iterator for image number.        

def frame_extraction(video_path , start_time , end_time, frame_index):
    print("=====================================================================================")
    print("Extracting Frames")
    print(video_path)
    Frame_path = 'C:/Users/VIJAY/FYP_project/webpage 1/static/Frame'

    if(os.path.isdir(Frame_path)):
        shutil.rmtree(Frame_path)
        print("Deleted Results dir")
        os.makedirs(Frame_path)
        print("Created ")
    else:
        os.makedirs(Frame_path)
        print("created new")

    #reading video
    cap = cv2.VideoCapture(video_path)
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # Set the start and end times in seconds
    # start_time = 0
    # end_time = 4

    # Calculate the start and end frame numbers
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")


    if num_frames > 1000:
        print("Input video too long")
    elif num_frames < 125:
        print("Input video too short")

    count=0
    i=0
    while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame):
        ret, frame = cap.read()
        if ret == True:
            if count in frame_index:
                path= Frame_path + '/Frames'+str(i)+'.jpg'
                cv2.imwrite(path,frame)
                print(path)
                i+=1
            count+=1
        else:
            break

    print(num_frames)
    #select the path 
    path = 'C:/Users/VIJAY/FYP_project/webpage 1/static/Frame'
    haarcascade(path)


def index_frame(file_path , start_frame_time , end_frame_time):
    
    #reading video
    video_path = file_path
    # Set the start and end times in seconds
    start_time = start_frame_time
    end_time = end_frame_time
    cap = cv2.VideoCapture(video_path)
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    if num_frames > 1000:
        print("Input video too long")
    elif num_frames < 125:
        print("Input video too short")

    frame_index = np.linspace(1,end_time * fps,180).astype(int)
    print("The index frames are to be extracted :\n",frame_index)
    frame_extraction(video_path , start_frame_time , end_frame_time, frame_index)










