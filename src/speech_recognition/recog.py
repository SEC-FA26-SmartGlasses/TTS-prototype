
import pyaudio
import openwakeword
import numpy as np
import silero_vad
from transformers import pipeline

CHUNK_SIZE = 512

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
    ww_model.reset()

    while True:
        audio = np.frombuffer(audio_stream.read(CHUNK_SIZE), dtype=np.int16)
        prediction = list(ww_model.predict(audio).values())[0]
        if prediction >= threshold:
            break

    return

def get_command_audio(audio_stream, vad_iterator):
    vad_iterator.reset_states()
    frames = []
    started = False
    while True:
        data = audio_stream.read(CHUNK_SIZE)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / (2**15)

        speech_event = vad_iterator(audio_chunk)
        if speech_event:
            if "start" in speech_event:
                started = True
            elif "end" in speech_event:
                break

        if started:
            frames.append(data)
    return b"".join(frames)



if __name__ == "__main__":
    # how to initialize:
    # initialize the voice models and microphone capture
    mic = init_mic_capture()

    ww_model = init_ww_model("hey_jarvis_v0.1")

    vad_model = silero_vad.load_silero_vad()
    vad_iterator = silero_vad.VADIterator(
        vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=700,
        speech_pad_ms=100,
    )

    # speech recognition model
    asr_model = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
    asr_model.model.generation_config.max_new_tokens = 1000
    
    while True:
        wait_for_wake_word(mic, ww_model)
        print("What is your command?")

        # this will extract one segment of audio
        # it will stop after 1 second of silence
        command_audio = get_command_audio(mic, vad_iterator) 
        audio_np = np.frombuffer(command_audio, dtype=np.int16)

        # feed audio back into ASR model
        out = asr_model({
            "raw": audio_np,
            "sampling_rate": 16000,
            }
        )

        print(out["text"])


  