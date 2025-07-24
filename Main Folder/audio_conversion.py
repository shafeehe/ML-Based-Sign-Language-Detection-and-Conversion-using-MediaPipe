import pyttsx3
import threading
import queue

#initialize text-to-speech engine
engine=pyttsx3.init()

#create a thread-safe queue (since pyttsx3 is not thread-safe, use QUEUE for THREAD SAFE HANDLING)
tts_queue = queue.Queue()

#function to process the queue
def process_queue():
    while True:
        #Get the next target from the queue
        predicted_sign = tts_queue.get()
        if predicted_sign is None:  #Exit signal
            break
        engine.say(predicted_sign)
        engine.runAndWait()
        tts_queue.task_done()

#start the processing thread
tts_thread = threading.Thread(target=process_queue, daemon=True)
tts_thread.start()


#function to convert text to speech
def sign_to_audio(predicted_sign):
    
    #(Adds the predicted hand sign to the queue for TTS conversion)

    tts_queue.put(predicted_sign)


#optional: Function to stop the TTS thread gracefully
def stop_tts():
    tts_queue.put(None)  #send exit signal
    tts_thread.join()