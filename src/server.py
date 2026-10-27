import speech_recognition.recog as sr

import socket
import cv2
import numpy as np
import struct
import re
import json

from paddleocr import PaddleOCR

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 51000

OCR_TRIGGERS = ["read", "read this", "text"]
TRIGGER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in OCR_TRIGGERS) + r")\b"
)

def send_json(sock, data):
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)

def recv_exact(sock, size):
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed before packet was complete")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)



def receive_command_packet(sock):
    header = recv_exact(sock, 8)
    frame_len, audio_len = struct.unpack("!II", header)

    # Validate payload sizes
    if not (0 < frame_len <= 20_000_000):
        raise ValueError(f"Invalid frame length: {frame_len}")
    if not (0 < audio_len <= 20_000_000):
        raise ValueError(f"Invalid audio length: {audio_len}")
    if audio_len % 2 != 0:
        raise ValueError("Invalid int16 PCM audio length")

    frame_bytes = recv_exact(sock, frame_len)
    audio_bytes = recv_exact(sock, audio_len)

    frame = cv2.imdecode(
        np.frombuffer(frame_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        raise ValueError("Failed to decode JPEG frame")

    audio_pcm = np.frombuffer(audio_bytes, dtype="<i2")

    return frame, audio_pcm

def main():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen()

    asr_model = sr.init_asr_model()

    paddle_ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_recognition_model_name="PP-OCRv6_tiny_rec",
        use_angle_cls=True, lang='en', enable_mkldnn=False
    )
    while True:
        client_sock, client_addr = server_socket.accept()
        with client_sock:
            print(f"Connected to {client_addr}")
            while True:
                try:
                    frame, audio = receive_command_packet(client_sock)
                    command = sr.process_asr(audio, asr_model)
                    print(command)

                    response = {}

                    response["command"] = command
                    response["tts_strings"] = []

                    if TRIGGER_PATTERN.search(command.lower()):
                        frame_results = paddle_ocr.predict(frame)
                        response["tts_strings"] = [res["rec_texts"] for res in frame_results]

                    print(response)

                    send_json(client_sock, response)

                except ConnectionError:
                    print(f"Client {client_addr} disconnected")
                    break




            
        


if __name__ == "__main__":
    main()