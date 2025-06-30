import asyncio
import os
import cv2
import json
import argparse
import numpy as np
import aiohttp
import sys  # <-- Agregado

from src.StreamCaptureService import StreamCaptureService
from src.DetectionService import DetectionService
from src.Tracker import Tracker
from src.InOutUseCase import InOutUseCase

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

# Servicios
streamCaptureService = StreamCaptureService(
    stream_url=stream['input']['url'],
    fps=stream['input']['fps'],
    dimensions=stream['input']['size']
)
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

# Sesión global aiohttp
session = None

async def callback(frame, config, stream):
    global session

    if session is None:
        session = aiohttp.ClientSession()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = *stream['roi'][0], *stream['roi'][1]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    frame_cropped = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
    response = await detectionService.execute(frame_cropped)
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

    frame_resized = cv2.resize(frame, (1000, 562))
    cv2.imwrite("output.jpg", frame_resized)
    _, img_encoded = cv2.imencode('.jpg', frame_resized)
    image_bytes = img_encoded.tobytes()

    try:
        backend_url = f"http://127.0.0.1:8000/processed_stream/{stream_id}"
        async with session.post(backend_url, data=image_bytes) as resp:
            await resp.read()  # Asegura que la conexión se libere
    except aiohttp.ClientError as e:
        print(f"Error sending processed frame for {stream_id}: {e}")
    except Exception as e:
        print(f"General error sending processed frame for {stream_id}: {e}")

async def main():
    try:
        await streamCaptureService.start_stream(lambda frame: callback(frame, config, stream))
    except Exception as e:
        print(f"Error crítico en el stream: {e}")
        sys.exit(1)  # 🔥 Forzar reinicio por systemd
    finally:
        if session:
            await session.close()
            print("Sesión HTTP cerrada correctamente.")

if __name__ == '__main__':
    asyncio.run(main())
