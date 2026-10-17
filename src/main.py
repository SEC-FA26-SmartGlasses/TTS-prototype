import speech_recognition.recog as sr
import text_to_speech.output_tts as tts
import text_extraction.image_process as im
import ocr.ocr as ocr
from paddleocr import PaddleOCR
import os

import re
import cv2
import threading

USE_PADDLE = False
crops_dir = "E:/test_outputs/"

stop_event = threading.Event()

latest_frame = None
captured_frame = None          # frame to show in the "Captured Frame" window
frame_lock = threading.Lock()

OCR_TRIGGERS = ["read", "read this", "text"]
TRIGGER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in OCR_TRIGGERS) + r")\b"
)


def capture_frame():
    with frame_lock:
        return None if latest_frame is None else latest_frame.copy()



def voice_loop():
    global captured_frame

    mic = sr.init_mic_capture()
    ww_model = sr.init_ww_model("hey_jarvis_v0.1")
    vad_iterator, asr_model = sr.init_asr_models()
    yolo_model = im.init_yolo_model("./src/models/text_extraction_model.pt")
    if USE_PADDLE:
        paddle_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_tiny_det",
            text_recognition_model_name="PP-OCRv6_tiny_rec",
            use_angle_cls=True, lang='en', enable_mkldnn=False
        )

    while not stop_event.is_set():
        sr.wait_for_wake_word(mic, ww_model)
        print("wake word found")
        print("listening!")
        command = sr.listen_for_command(mic, vad_iterator, asr_model)
        print(f"Read command: {command}")

        if TRIGGER_PATTERN.search(command.lower()): # OCR trigger
            print("Performing OCR!")
            frame = capture_frame()
            if frame is None:
                print("No camera frame available yet")
                tts.output_audio("I can't see anything yet")
                continue

            with frame_lock:
                captured_frame = frame



            output_strings = []

            if USE_PADDLE:
                results = paddle_ocr.predict(frame)
                for r in results:
                    output_strings.extend(r["rec_texts"])
            else:
                processed_crops = im.process_image(frame, yolo_model, 50)
                # print(processed_crops)

                sorted_crops = sorted(
                    processed_crops.values(),
                    key=lambda crop: crop["coordinates"][1]
                )
                output_strings = ocr.img_to_text(sorted_crops)
                # output_strings = ocr.img_to_text([{"processed_crop": frame}])
                cv2.imwrite(os.path.join(crops_dir, f"frame.png"), frame)
                for i, crop in enumerate(sorted_crops):
                    crop_image = crop["processed_crop"]

                    if crop_image is not None:
                        filename = os.path.join(crops_dir, f"crop_{i:03d}.png")
                        cv2.imwrite(filename, crop_image)
                        print(f"Saved crop: {filename}")

            for str in output_strings:
                print(str)

            for str in output_strings:
                if any(char.isalnum() for char in str):
                    tts.output_audio(str)


def main():

    # The opencv webcam preview and such has to be on the main thread
    # essentially I always update the webcam/latest frame
    
    # so I end up having to move the voice command loop to another thread
    # we could rearrange this if we don't want a constant webcam preview

    global latest_frame, captured_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            with frame_lock:
                latest_frame = frame
                snapshot = captured_frame
                captured_frame = None

            cv2.imshow("Webcam", frame)
            if snapshot is not None:
                cv2.imshow("Captured Frame", snapshot)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("exiting")
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()