from gtts import gTTS
import pyaudio
from io import BytesIO
from pydub import AudioSegment

#function to setup gtts
def setup_google_tts(text):

    tts = gTTS(text=text, lang="en", slow=False)
    
    return tts

# function to convert string to audio
def string_to_audio(text):
    tts = setup_google_tts(text)

    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    audio = AudioSegment.from_file(audio_buffer, format="mp3")

    return audio


#function to output audio
def output_audio(text):
    audio = string_to_audio(text)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True,
        frames_per_buffer=1024
    )

    try:
        audio_data = audio.raw_data

        for i in range(0, len(audio_data), 1024):
            stream.write(audio_data[i:i + 1024])

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    output_audio(text)