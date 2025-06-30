import subprocess
import numpy as np
import cv2
import time
import threading
import argparse
import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from src.DetectionService import DetectionService
from src.Tracker import Tracker
from src.InOutUseCase import InOutUseCase

load_dotenv()

FILE_PATH = '/var/lib/vhs'

# Argumentos
parser = argparse.ArgumentParser(description='VHS Camera Stream Service')
parser.add_argument('--stream-id', type=str, required=True, help='ID del stream a procesar')
args = parser.parse_args()
stream_id = args.stream_id

# Leer configuración
config_path = os.path.join(FILE_PATH, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
    stream = next((s for s in config['streams'] if s['id'] == stream_id), None)

    if stream is None:
        raise ValueError(f"No se encontró la transmisión con ID {stream_id}")


stream_url  = stream['input']['url']

width       = stream['input']['size'][0]
height      = stream['input']['size'][1]
fps         = stream['input']['fps']

new_width   = 640
new_height  = 360

frame_size = width * height * 3  # bytes por frame BGR24

detectionService = DetectionService()
use_case = InOutUseCase(
    stream=stream,
    tracker=Tracker(
        min_threshold=stream['tracker']['min_threshold'],
        max_threshold=stream['tracker']['max_threshold'],
        max_lost_frames=stream['tracker']['max_lost_frames']
    ),
    counter_interpolation=stream['counter_interpolation'],
    cross_line=stream['detection_cross_line'],
    centroid_orientation=stream['centroid_orientation']
)

def read_stderr(pipe):
    """Lee el stderr sin bloquear para evitar que FFmpeg se quede esperando."""
    while True:
        line = pipe.readline()
        if not line:
            break
        print("FFmpeg stderr:", line.decode().strip())

def start_ffmpeg_process():
    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-stimeout", "15000000",
        "-i", stream_url,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-"
    ]
    return subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

max_retries = 5
retry_delay = 5  # segundos

retries = 0
saved = False

while retries < max_retries:
    print(f"Intento #{retries + 1} de conexión al stream")
    process = start_ffmpeg_process()

    # Lanzar hilo para leer stderr sin bloquear
    stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))
    stderr_thread.daemon = True
    stderr_thread.start()

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)

            if not raw_frame:
                print("Stream terminado o no se reciben más datos")
                break

            if len(raw_frame) < frame_size:
                print(f"Frame incompleto recibido ({len(raw_frame)} bytes), terminando lectura")
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()

            """
                Aqui comienza la logica de procesamiento del frame
            """

            x1, y1, x2, y2 = *stream['roi'][0], *stream['roi'][1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # recortar el frame si aplica
            frame_cropped = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
            
            # Deteccion de objetos
            response = asyncio.run(detectionService.execute(frame_cropped))
            print(response)
            
            frame_cropped = use_case.execute(
                frame=frame_cropped,
                place={"code": config['code'], "name": config['name']},
                detections=response['results']
            )

            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = frame_cropped
            else:
                frame = frame_cropped
                
            frame_resized = cv2.resize(frame, (new_width, new_height))
            cv2.imwrite(f"{stream['id']}-output.jpg", frame_resized)
            
            frame = frame_resized
            
            """
                Fin de procesamiento del frame
            """
            
            if(os.getenv("WINDOW_ENABLED")):
                cv2.imshow('prueba', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrumpido por usuario")
                process.terminate()
                process.wait()
                exit(0)

    except Exception as e:
        print(f"Error leyendo frames: {e}")

    finally:
        process.terminate()
        process.wait()
        print("Proceso FFmpeg terminado")

    retries += 1
    print(f"Reconectando en {retry_delay} segundos...")
    time.sleep(retry_delay)

print("No se pudo conectar al stream después de varios intentos. Saliendo.")
cv2.destroyAllWindows()
exit(0)