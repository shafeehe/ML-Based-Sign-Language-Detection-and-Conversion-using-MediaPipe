import tkinter as tk
from PIL import Image, ImageTk
import cv2
import speech_recognition as sr
import os
import threading

from audio_conversion import sign_to_audio,stop_tts

#global flag to control audio-to-handsign conversion
stop_audio_to_handsign = False
tkinter_window = None #global variable to store the tkinter window
tkinter_thread = None #global variable to store the tkinter thread

# Mapping dictionary for hand signs
hand_sign_mapping = {
    "stop": {"image": "images/stop.png"},
    "close": {"image": "images/close.png"},
    "give" : {"gif": "gifs/give.gif"},
    "i need help" : {"gif": "gifs/i need help.gif"},
    "i am sorry" : {"gif": "gifs/i am sorry.gif"},
    "i need food" : {"gif":"gifs/to eat.gif"},
    "what is your name" : {"gif": "gifs/what is your name.gif"},
    "where" : {"gif": "gifs/where.gif"},
    "why" : { "gif": "gifs/why.gif"},
    "i want" : {"gif": "gifs/i want.gif"},
    "we love you": {"gif": "gifs/we love you.gif", "video": "videos/we love you.mp4"},
    "where is thebathroom" : {"gif": "gifs/where is the bathroom.gif"},
    "how are you" : {"gif": "gifs/how are you.gif"}
}

# Function to display hand sign (Image or GIF)
def display_hand_sign(media_paths):
    global tkinter_window, tkinter_thread

    if "video" in media_paths and os.path.exists(media_paths["video"]):
        play_video(media_paths["video"])
    else:
        def tkinter_main():
            global tkinter_window

            tkinter_window = tk.Tk()
            tkinter_window.title("Hand Sign Display")
            tkinter_window.geometry("500x500")
            tkinter_window.configure(bg='lightgray')

            label_text = tk.Label(tkinter_window, text="Recognized Sign:", font=("Arial", 16), bg='lightgray')
            label_text.pack()

            label = tk.Label(tkinter_window)  # Empty label for the image
            label.pack()

            frames = []  # Initialize the list of frames

            if "gif" in media_paths and os.path.exists(media_paths["gif"]):
                gif = Image.open(media_paths["gif"])
                
                # Instead of creating PhotoImage objects here, we store the raw frames
                for i in range(gif.n_frames):
                    gif.seek(i)
                    frames.append(gif.copy().convert("RGBA"))  # Store images as PIL objects

                def process_gif_frames():
                    """ Converts PIL frames into ImageTk inside the Tkinter thread. """
                    processed_frames = [ImageTk.PhotoImage(frame) for frame in frames]
                    animate_gif(processed_frames)

                def animate_gif(processed_frames, index=0):
                    if not stop_audio_to_handsign:
                        label.config(image=processed_frames[index])
                        tkinter_window.after(100, animate_gif, processed_frames, (index + 1) % len(processed_frames))
                    else:
                        tkinter_window.destroy()

                # Convert and animate frames after Tkinter is fully initialized
                tkinter_window.after(500, process_gif_frames)

            elif "image" in media_paths and os.path.exists(media_paths["image"]):
                img = Image.open(media_paths["image"]).resize((300, 300))
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo)
                label.image = photo

            else:
                label.config(text="No valid image/GIF found", font=("Arial", 14), fg='red', bg='lightgray')

            exit_button = tk.Button(tkinter_window, text="Exit", command=tkinter_window.destroy, font=("Arial", 12))
            exit_button.pack(pady=10)

            tkinter_window.mainloop()
            tkinter_window = None  # Reset global variable

        tkinter_thread = threading.Thread(target=tkinter_main, daemon=True)
        tkinter_thread.start()



# Function to play video with integrated UI buttons
def play_video(video_path):
    global stop_audio_to_handsign

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): # Check if video file is opened
        print(f"Error: Could not open video file: {video_path}")
        return
    
    cv2.namedWindow("Hand Sign Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Sign Video", 640, 480)  # Resize window

    paused = False
    def toggle_pause():
        nonlocal paused
        paused = not paused
    
    
    def exit_video():
        cap.release()
        cv2.destroyAllWindows()

    while cap.isOpened() and not stop_audio_to_handsign:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Auto replay
                continue
            
            # Add buttons directly to the video frame
            cv2.putText(frame, "Press P - Pause/Play", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press R - Replay", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC - Exit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Hand Sign Video", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Press ESC to exit video
            break
        elif key == ord('p'):  # Press 'p' to pause/play
            toggle_pause()

    
    cap.release()
    cv2.destroyAllWindows()

# Speech recognition function
def recognize_speech():
    global stop_audio_to_handsign

    recognizer = sr.Recognizer()

    sign_to_audio("Now you can speak")
    try:

        with sr.Microphone() as source:
            print("Listening for audio input...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source) 
                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized Text: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Error with speech recognition: {e}")
    except AttributeError as e:
        print(f"Microphone access Error: {e}")
    except Exception as e:
        print(f"unexpected Error in recognize_speech: {e}")

    return None

# Main function to handle recognized speech
def audio_to_hand_sign(audio_to_handsign_enabled):
    global stop_audio_to_handsign

    while audio_to_handsign_enabled and not stop_audio_to_handsign:
        recognized_text = recognize_speech()
        if recognized_text is None:
            print("No valid input detected.")
            continue

        if recognized_text in hand_sign_mapping:
            media_paths = hand_sign_mapping[recognized_text]
            print(f"Displaying hand sign for: {recognized_text}")

            if "video" in media_paths and os.path.exists(media_paths["video"]):
                play_video(media_paths["video"])
            else:
                display_hand_sign(media_paths)
        else:
            print(f"No hand sign mapped for: {recognized_text}")

if __name__ == "__main__":
    print("Starting Audio-to-Handsign System...")
    audio_to_hand_sign(audio_to_handsign_enabled=True)  #default value for testing
