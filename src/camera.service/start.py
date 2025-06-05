from src.StreamCaptureService import StreamCaptureService
from src.DetectionService import DetectionService
from src.Tracker import Tracker
from src.InOutUseCase import InOutUseCase
import asyncio, os, cv2, time, json, argparse
import numpy as np
import aiohttp

# Obtener el stream_id desde los argumentos de línea de comandos
parser = argparse.ArgumentParser(description='VHS Camera Stream Service')
parser.add_argument('--stream-id', type=str, required=True, help='ID del stream a procesar')
args = parser.parse_args()
stream_id = args.stream_id

stream = None
config_path = os.path.join('/var/lib/vhs', 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
    for s in config['streams']:
        if s['id'] == stream_id:
            stream = s
            break
    if stream is None:
        raise ValueError(f"No se encontró la transmisión con ID {stream_id}")

# Inicialización de servicios
streamCaptureService = StreamCaptureService(stream_url=stream['input']['url'], fps=stream['input']['fps'], dimensions=stream['input']['size'])
detectionService = DetectionService()
use_case = InOutUseCase(
    tracker=Tracker(min_threshold=stream['tracker']['min_threshold'], max_threshold=stream['tracker']['max_threshold'], max_lost_frames=stream['tracker']['max_lost_frames']),
    counter_interpolation=stream['counter_interpolation'],
    cross_line=stream['detection_cross_line'],
    centroid_orientation=stream['centroid_orientation']
)

async def callback(frame, config, stream):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = *stream['roi'][0], *stream['roi'][1]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Validar y recortar solo si es necesario
    if x2 > x1 and y2 > y1:
        frame_cropped = frame[y1:y2, x1:x2]
    else:
        frame_cropped = frame
        
    response = await detectionService.execute(frame_cropped)
    print(response)
    frame_cropped = use_case.execute(frame=frame_cropped, place={"code": config['code'], "name": config['name']}, detections=response['results'])
    
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = frame_cropped
    else:
        frame = frame_cropped

    frame_resized = cv2.resize(frame, (1000, 562))
    _, img_encoded = cv2.imencode('.jpg', frame_resized)
    image_bytes = img_encoded.tobytes()

    async with aiohttp.ClientSession() as session:
        try:
            backend_url = "http://127.0.0.1:8000/processed_stream/" + stream_id
            await session.post(backend_url, data=image_bytes)
        except aiohttp.ClientError as e:
            print(f"Error sending processed frame for {stream_id}: {e}")
        except aiohttp.ClientTimeout as e:
            print(f"Timeout sending processed frame for {stream_id}: {e}")
        except Exception as e:
            print(f"General error sending processed frame for {stream_id}: {e}")

async def main():
    await streamCaptureService.start_stream(lambda frame: callback(frame, config, stream))

if __name__ == '__main__':
    asyncio.run(main())