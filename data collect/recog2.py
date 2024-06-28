import cv2
import dlib
import os
import tkinter as tk
from tkinter import simpledialog

# Create a folder to store captured face images
if not os.path.exists('faces'):
    os.makedirs('faces')

# Start video capture for webcam -  Specifying 0 as an argument fires up the webcam feed
video_capture = cv2.VideoCapture(0)

# Load the pre-trained face detector model
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Counter for naming saved face images
counter = 0

def capture_face():
    global counter
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Detect faces in the frame
    dets = cnn_face_detector(rgb_small_frame, 1)
    
    for det in dets:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top = 4 * det.rect.top()
        right = 4 * det.rect.right()
        bottom = 4 * det.rect.bottom()
        left = 4 * det.rect.left()
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Display a pop-up window to enter the entity name
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        entity_name = simpledialog.askstring("Entity Name", "Enter the name for this entity:")
        
        # Save the detected face as an image with the entered entity name
        face_image = frame[top:bottom, left:right]
        cv2.imwrite(f'faces/{entity_name}_{counter}.jpg', face_image)
        counter += 1
        
        root.destroy()  # Close the pop-up window
        
        # Label the face as "Unknown"
        cv2.putText(frame, 'Unknown', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imshow('Dlib_CNN', frame)

while True:
    # Check if 'a' key is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('a'):
        capture_face()
    
    # Hit 'q' on the keyboard to quit!
    if key & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
