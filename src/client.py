import cv2
import threading
import struct
import socket
import speech_recognition.recog as sr
import text_to_speech.output_tts as tts


import argparse
import json


SERVER_PORT = 9999

stop_event = threading.Event()

latest_frame = None
captured_frame = None          # frame to show in the "Captured Frame" window
frame_lock = threading.Lock()


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


def receive_json(sock):
    header = recv_exact(sock, 4)
    payload_len = struct.unpack("!I", header)[0]

    if not (0 < payload_len <= 1_000_000):
        raise ValueError("Invalid JSON response length")

    payload = recv_exact(sock, payload_len)
    return json.loads(payload.decode("utf-8"))


def send_command_to_server(sock, frame, audio_pcm):

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("failed to encode frame")

    frame_bytes = encoded.tobytes()

    header = struct.pack(
        "!II", len(frame_bytes), len(audio_pcm.tobytes())
    )

    sock.sendall(header)
    sock.sendall(frame_bytes)
    sock.sendall(audio_pcm)

    return

def capture_frame():
    with frame_lock:
        return None if latest_frame is None else latest_frame.copy()


def client_loop(host):
    global captured_frame

    server_host = host

    mic = sr.init_mic_capture()
    ww_model = sr.init_ww_model("hey_jarvis_v0.1")
    vad_iterator = sr.init_vad_model()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((server_host, SERVER_PORT))
            print(f"Connected to server at {server_host}:{SERVER_PORT}")
            while not stop_event.is_set():
                sr.wait_for_wake_word(mic, ww_model)
        
                frame = capture_frame()
                if frame is None:
                    print("No camera frame available yet")
                    tts.output_audio("I can't see anything yet")
                    continue
                with frame_lock:
                    captured_frame = frame
        
        
                print("wake word found")
                print("listening!")
        
                audio = sr.listen_for_command(mic, vad_iterator)

                send_command_to_server(sock, frame, audio)
                response = receive_json(sock)
                print(response)
                
                continue
        except ConnectionRefusedError:
            print("Could not connect to server!")


    

def main():

    parser = argparse.ArgumentParser(
        description="Voice-controlled webcam client"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server hostname or IP address (default: 127.0.0.1)",
    )
    args = parser.parse_args()

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

    voice_thread = threading.Thread(target=client_loop, args=(args.host,), daemon=True)
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