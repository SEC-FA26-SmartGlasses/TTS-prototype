# TTS Prototype

This repo contains the source code for OCR/TTS prototype. To run, install the dependencies in requirements.txt in a virtual environment, and install:

- https://ffmpeg.org/download.html
- https://tesseractocr.org/install

And ensure it is added to PATH. 

The project is organized as follows:

- src/main.py: the main control flow and manages the camera and processing 
- src/speech_recognition/recog.py: speech-to-text command pipeline 
- src/text_extraction/image_process.py: image preprocessing and text detection algorithms to use before OCR
- src/ocr/ocr.py: wrapper around tesseractocr that takes the processed image format as input.
- src/text_to_speech/output_tts.py: wrappers around Google TTS to use for outputting audio to speakers

Currently, the ocr.py and image_process.py are unused, as we are unable to get good performance out of the model. We have instead been using PaddleOCR which doesn't require any preprocessing.
