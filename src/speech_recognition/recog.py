
import pyaudio
import openwakeword
import numpy as np

CHUNK_SIZE = 1280

def init_mic_capture():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    audio = pyaudio.PyAudio()

    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    print("Listening on:", audio.get_default_input_device_info()["name"])
    return mic_stream

def init_ww_model(model_path):
    openwakeword.utils.download_models(model_names=[model_path]) # if we train our own model replace this
    model = openwakeword.Model(wakeword_models=[model_path], inference_framework="onnx")
    return model

def wait_for_wake_word(audio_stream, ww_model, threshold=0.7):
    while True:
        audio = np.frombuffer(audio_stream.read(CHUNK_SIZE), dtype=np.int16)
        prediction = list(ww_model.predict(audio).values())[0]
        if prediction >= threshold:
            break
        



# while testing, try to keep everything inside of a block like this
# these won't get called when the module is imported but it will be called
# if you try to run the file
if __name__ == "__main__":
    chunk_size = 1280
    mic = init_mic_capture() # this will use default device
    ww_model = init_ww_model("hey_jarvis_v0.1")
    wait_for_wake_word(mic, ww_model)
    print("hi!")