#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque


import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import threading

from audio_conversion import sign_to_audio, stop_tts
from audio_to_handsign import audio_to_hand_sign
from audio_to_handsign import tkinter_window, tkinter_thread, stop_audio_to_handsign

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

#Global flag for audio-to-handsign conversion
audio_to_handsign_enabled = False

current_class_id = -1  # Tracks the first key press in a two-key combination

def toggle_audio_to_hand_sign():
    global audio_to_handsign_enabled
    audio_to_handsign_enabled = not audio_to_handsign_enabled
    print(f"Audio to HandSign Conversion {'Enabled' if audio_to_handsign_enabled else 'Disabled'}")

    #start a new thread for audio-to-handsign conversion if enabled
    if audio_to_handsign_enabled:
        threading.Thread(target=audio_to_hand_sign, args=(audio_to_handsign_enabled,), daemon=True).start()

        
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


#Adding Mode toggling to switch between modes. (newly added mode = 3 to select mode 'Audio to handsign')
def process_mode (key, mode, number, landmark_list, point_history_list):
        global audio_to_handsign_enabled

        if mode == 0: #default/placeholder
            pass
        elif mode ==2: #logging point history
            if 0 <= number <= 9:
                logging_csv(number, mode, landmark_list, point_history_list)
        elif mode == 3: #Audio to Handsign conversion
            if not audio_to_handsign_enabled:
                toggle_audio_to_hand_sign() #toggle the audio-to-handsign conversion ON/OFF
            mode = 0 #reset mode to default after toggling

        return mode

##################Function to Implement gesture speed tracking to improve recognition realism ############################



#function to calculate speed of reference finger(tip of index finger here with landmark 8)
def calculate_speed(prev_landmark, current_landmark, time_elapsed):
    if prev_landmark is None or current_landmark is None:
        return 0  #default speed if no landmark is detected.
    
    distance = np.linalg.norm(np.array(current_landmark) - np.array(prev_landmark))
    speed = distance / time_elapsed #speed = distance/time
    return speed



def main():
    global cap, stop_audio_to_handsign

    
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW) #Use DirectShow for faster capture
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    keypoint_classifier_labels = []
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                keypoint_classifier_labels.append(row[0])


    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    #changes in app.py when using thread to prevent fps drop & delays
    last_predicted_sign = None
    last_tts_time = 0
    tts_delay = 1 #Minimum delat b/w TTS calls

    #initializing the variables for gesture speed tracking
    prev_landmark = None #stores the landmark (of index finger) of previous frame
    prev_time = time.time() #stores the time of the previous frame


    sign_sentence = [] #Buffer to store recognized signs
    last_sign_time = time.time() #time of the last recognized sign
    sentence_delay = 3 #time after which a sentence is finalized                     

    sentence_display = "" #announced sentence is stored for display

    announce_enabled = True #flag to enable/disable TTS announcement

    try:
        while True:
            fps = cvFpsCalc.get()

            #default value for predicted sign
            predicted_sign = None  #prevents the error of calling 'sign_to_audio' with None value(UnboundedLocalError)

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                stop_audio_to_handsign = True
                break
            elif key == ord('c'): #assigning 'c' to clear the sentence buffer
                sign_sentence.clear()
                sentence_display = "" #clear displayed sentence
                last_predicted_sign = None #clear last predicted sign


            elif key == ord('s'): #assigning 's' to stop the sentence formation
                last_sign_time = time.time() - sentence_delay #force the sentence to be finalized
                print("Sentence formation manually stopped by user")

            elif key == ord('a'): #assigning 'a' to toggle TTS announcement
                announce_enabled = not announce_enabled #toggle state ON/OFF
                print(f"TTS Announcement {'is turned ON' if announce_enabled else 'is turned OFF'}")


            number, mode = select_mode(key, mode)


            # Camera capture frame-by-frame #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation(convert image for Mediapipe processing) #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #default values for landmark and point history(for solving the error of prematurely calling 'process_mode' inside main loop)
            landmark_list=[]
            point_history_list=[]


            #  ####################################################################
            if results.multi_hand_landmarks is not None:  #(when no hands detected)
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)


    ############Gesture Speed Tracking Implementation######################

                    #track gesture speed
                    current_time = time.time() 
                    time_elapsed = current_time - prev_time     #time diff b/w last frame time and current frame time
                    current_landmark = landmark_list[8] #landmark of index fingertip

                    speed = calculate_speed(prev_landmark, current_landmark, time_elapsed) #compute speed
                    prev_landmark = current_landmark
                    prev_time = current_time

                    #adjust confidence threshold based on gesture speed
                    if speed > 0.1: #if user is signing too fast, higher the confidence threshold, reducing false positives
                        confidence_threshold = 0.88 
                    else: 
                        confidence_threshold = 0.65 #if user signing slow, high confidence threshold

                    print(f"Gesture Speed: {speed:.4f} | Confidence Threshold: {confidence_threshold:.2f}")

                    #########################################################################################

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id, confidence = keypoint_classifier(pre_processed_landmark_list)
                    confidence_threshold = 0.65

                    #Only proceed if confidence is above a certain threshold ----> modification: to Add a Cooldown Period to Reduce Fluctuations(false positives) in b/w.
                    if confidence > confidence_threshold and (time.time() - last_sign_time > 0.5): #cool down time of 0.5s to prevent immediate re-triggering of gesture recognition
                        if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                            predicted_sign = keypoint_classifier_labels[hand_sign_id]
                        else:
                            predicted_sign = "Unknown"

                  


                        #adding words to sentence buffer if its different from the last word
                        if predicted_sign != 'Unknown' and predicted_sign != last_predicted_sign:
                            if time.time() - last_tts_time > tts_delay: #minimum delay between TTS calls
                                sign_sentence.append(predicted_sign) #adding sign to the sentence buffer
                                last_sign_time = time.time()
                                print(f"added '{predicted_sign}' to sentence: {' '.join(sign_sentence)}")

                    
                    else:
                        print(f"Low confidence: Gesture recognition skipped (Confidence: {confidence:.2f})")    

                    if hand_sign_id == 'None':  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])


                    if predicted_sign is None: 
                        predicted_sign = "Unknown" #default value for predicted sign when no sign is detected

                    #calling the TTS function to announce the gesture (through threading to make TTS asynchronous)
                    current_time = time.time()
                    if predicted_sign != "Unknown" and predicted_sign != last_predicted_sign and (current_time - last_tts_time > tts_delay): #"unknown" is not announced
                        if announce_enabled:
                            sign_to_audio(predicted_sign)
                            last_predicted_sign = predicted_sign
                            last_tts_time = current_time   

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()
                
                

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        predicted_sign,
                        keypoint_classifier_labels[hand_sign_id],
                        #point_history_classifier_labels[most_common_fg_id[0][0]],
                    )

                    #convert predicted handsign to speech
                    #sign_to_audio(predicted_sign)--->NOT NEEDED HERE
            else:
                point_history.append([0, 0])

    
        #################"Sentence" announcement Logic######################
            if time.time() - last_sign_time > sentence_delay and len(sign_sentence) > 0:
                sentence = " ".join(sign_sentence)  #convert list to full sentence

                #avoid announcing single-word sentences unless necessary
                if len(sign_sentence) >1:
                    if announce_enabled:

                        sign_to_audio(sentence)
                        sentence_display = sentence #display the sentence on the screen
            
                sign_sentence.clear() #Reset buffer after speaking the sentence
                last_sign_time = time.time() #reset the last sign time to prevent immediate re-triggering of sentence announcement
        

            #call the process_mode function (update in main loop when integrating audio to handsign{adding mode toggling})
            process_mode(key, mode, number, landmark_list, point_history_list)

            # Screen reflection #############################################################
            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)
            cv.putText(debug_image, f"Sentence: {sentence_display}", (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA) #displaying sentence
            cv.putText(debug_image, f"Announcing: {'is ON' if announce_enabled else 'is OFF'}", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA) #displaying TTS status
            cv.imshow('Hand Gesture Recognition', debug_image)
    except KeyboardInterrupt:
        print("program terminated by user")
    finally:
        if 'cap' in globals():
            cap.release()
        cv.destroyAllWindows()
        stop_tts()  #stops the TTS processing thread(solving the RuntimeError:run loop already started. Also preventing FPS drop using 'def process_queue' method)
        stop_audio_to_handsign = True #signal the audio-to-handsign thread to stop

        #explicitly destroy all windows to prevent the program from hanging
        if tkinter_window is not None:
            tkinter_window.destroy() 
        if tkinter_thread is not None:
            tkinter_thread.join()


def select_mode(key, mode):
    global current_class_id


    
    if 48 <= key <= 57:  # 0 ~ 9
        if current_class_id == -1:
            current_class_id = key - 48 #store the first key press
        else:
            #combine the first and second key presses to form a two-digit class ID

            number = current_class_id * 10 + (key - 48)
            current_class_id = -1 #reset after logging the class ID
            return number, mode
        
    
    if key == 110:  # n for Audio to Handsign
        mode = 3
    if key == 107:  # k for logging keypoints
        mode = 1
    if key == 104:  # h
        mode = 2
    return -1, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 99):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    try: #try-except block to handle KeyboardInterrupt
        main()
    except KeyboardInterrupt:
        print("Program terminated by the user.")
    finally:
        if 'cap' in globals():
            cap.release()
        cv.destroyAllWindows()
        stop_tts() #stop the TTS thread

