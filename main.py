import face_recognition
import os, sys
import cv2
import numpy as np
import math

import pygame
import re

import datetime
import time
import threading

keep_running = True

# untuk face landmark untuk menampilkan kerangka frame pada wajah
# import itertools
# import numpy as np
# from time import time
# import mediapipe as mp
# import matplotlib.pyplot as plt

# Initialize the pygame mixer
pygame.mixer.init()

# Load the ring sound file
pygame.mixer.music.load("./ringbell/mixkit-old-telephone-ring-1357.mp3")

# Set the volume (optional)
pygame.mixer.music.set_volume(0.7)  # Volume level between 0.0 and 1.0
def play_ring_sound():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def face_confidence_percentage(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    face_rec_color = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):

        # Home owner Faces
        directory = 'faces'
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):  # Check if it's a directory
                print(f"Processing directory: {subdir}")
                for image_name in os.listdir(subdir_path):
                    # Check if the file is an image
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.HEIC')):
                        # Create a full path to the image
                        image_path = os.path.join(subdir_path, image_name)
                        # Load the image
                        face_image = face_recognition.load_image_file(image_path)
                        # Attempt to find face encodings in the image
                        face_encodings = face_recognition.face_encodings(face_image)
                        if face_encodings:
                            # Take the first face encoding found (assuming one face per image)
                            face_encoding = face_encodings[0]
                            self.known_face_encodings.append(face_encoding)
                            # Append the name or a part of it as a label for the encoding
                            self.known_face_names.append(os.path.splitext(image_name)[0])
                        else:
                            print(f"No faces found in the image {image_name}")
                    else:
                        print(f"Skipped non-image file: {image_name}")

        print("Loaded faces:", self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        iter_time = 0
        user_init = 0

        while True:
            ret, frame = video_capture.read()
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                self.face_rec_color = []

                count_face = 0
                iter_time = iter_time + 1
                rec_color = (0,0,0)
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 'Unknown'
                    conf_percent = 0
                    user = '???'
                    count_face = count_face + 1

                    # print(count_face)

                    print(f'Iter time : {iter_time}', end="\n")
                    print(f'UserInit: {user_init}', end="\n")
                    print("==================", end="\n")

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        conf_percent = face_confidence_percentage(face_distances[best_match_index])

                    if conf_percent >= 95 :
                        user = "Home Owner (Unlocked)"
                        user_init = 1
                        iter_time = 0
                        rec_color = (255, 0, 0)

                    else:
                        user = "Stranger (locked)"
                        user_init = 0
                        rec_color = (0, 0, 255)

                    stranger_time_to_alarm = 4
                    if user_init == 0 and iter_time > stranger_time_to_alarm :
                        play_ring_sound()

                    name = re.sub(r'\d', '', name)

                    self.face_names.append(f'{user} : {name} ({confidence})')
                    self.face_rec_color.append(rec_color)

            self.process_current_frame = not self.process_current_frame

            # display annotations
            for (top, right, bottom, left), name, rec_color in zip(self.face_locations, self.face_names, self.face_rec_color):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), rec_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), rec_color, -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

                cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image = face_recognition.load_image_file("./faces/roland/roland01.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)

    # print(face_landmarks_list)

    fr = FaceRecognition()
    fr.run_recognition()


