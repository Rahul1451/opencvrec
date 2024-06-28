import cv2
import face_recognition
import os
import pickle
import tkinter as tk
from tkinter import simpledialog
import numpy as np

# Function to save known faces
def save_known_faces(known_faces, filename="known_faces.pkl"):
    try:
        with open(filename, "wb") as f:
            pickle.dump(known_faces, f)
    except Exception as e:
        print(f"Error saving known faces: {e}")

# Function to load known faces
def load_known_faces(filename="known_faces.pkl"):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading known faces: {e}")
    return {}

# Function to prompt for a new name using tkinter
def prompt_for_name():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    root.destroy()
    return name

# Function to save a face image
def save_face_image(frame, face_location, name, index):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    os.makedirs(f"saved_faces/{name}", exist_ok=True)
    image_path = os.path.join(f"saved_faces/{name}", f"{index}.jpg")
    cv2.imwrite(image_path, face_image)

# Load known faces from the "faces" directory and merge with previously saved faces
known_faces = load_known_faces()  # Load previously saved faces from file
if not isinstance(known_faces, dict):
    known_faces = {}

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Set camera quality
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height
video_capture.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness
video_capture.set(cv2.CAP_PROP_CONTRAST, 50)  # Adjust contrast

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reduce frame size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Compare face encodings with known faces
        for known_name, known_encodings in known_faces.items():
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                name = known_name
                break

        if name == "Unknown":
            # Display a message prompting to press "a" to add a new name
            cv2.putText(frame, "Unknown face detected. Press 'a' to add name.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw a rectangle around the face
        left *= 2
        top *= 2
        right *= 2
        bottom *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        # Prompt for a new name if the face is unknown
        name = prompt_for_name()
        if name:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                if not any(face_recognition.compare_faces([e for encodings in known_faces.values() for e in encodings], face_encoding)):
                    if name not in known_faces:
                        known_faces[name] = []
                    known_faces[name].append(face_encoding)
                    save_known_faces(known_faces)
                    index = len(known_faces[name])
                    save_face_image(frame, (top*2, right*2, bottom*2, left*2), name, index)
                    print(f"New face added for {name}")

    # Press 'q' to quit
    if key == ord("q"):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
