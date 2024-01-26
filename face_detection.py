import cv2
import sys
import numpy as np
import os 

# Path to the Haar Cascade XML file
haar_cascade_file = r'C:\Users\shrut\OneDrive\Desktop\PROJECTS\face detection\haarcascade_frontalface_default.xml'
  
# Directory where all the face data will be saved
data_directory = 'datasets'  

# Name of the subdirectory for the current user
user_name = 'shruti'     

# Combining paths
user_directory = os.path.join(data_directory, user_name)

# Creating directories if they don't exist
if not os.path.exists(data_directory):
    os.mkdir(data_directory)
    
if not os.path.isdir(user_directory): 
    os.mkdir(user_directory) 

# Size of the images
image_width, image_height = 130, 100

# Initializing the face cascade classifier
face_cascade = cv2.CascadeClassifier(haar_cascade_file) 

# Accessing the webcam (webcam index '0' for default webcam)
webcam = cv2.VideoCapture(0)  

# The program loops until it has 30 images of the face. 
image_count = 1
while image_count < 15:  
    _, frame = webcam.read() 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray_frame[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (image_width, image_height)) 
        cv2.imwrite('%s/%s.png' % (user_directory, image_count), face_resize) 
        image_count += 1
      
    cv2.imshow('OpenCV', frame) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
