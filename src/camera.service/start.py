from src.StreamCaptureService import StreamCaptureService
from src.DetectionService import DetectionService
from src.Tracker import Tracker
from src.InOutUseCase import InOutUseCase
import asyncio, os, cv2, time, json
import numpy as np

config_path = os.path.join('/var/lib/vhs', 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Inicialización de servicios
streamCaptureService    = StreamCaptureService(stream_url=config['input']['url'], fps=config['input']['fps'], dimensions=config['input']['size'])
detectionService        = DetectionService()
use_case                = InOutUseCase(
    tracker=Tracker(min_threshold=config['tracker']['min_threshold'], max_threshold=config['tracker']['max_threshold'], max_lost_frames=config['tracker']['max_lost_frames']),
    counter_interpolation=config['counter_interpolation'],
    cross_line=config['detection_cross_line'],
    centroid_orientation=config['centroid_orientation']
)

async def callback(frame, config):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = *config['roi'][0], *config['roi'][1]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Validar y recortar solo si es necesario
    if x2 > x1 and y2 > y1:
        frame_cropped = frame[y1:y2, x1:x2]
    else:
        frame_cropped = frame
    
    frame_cropped = cv2.resize(frame_cropped, (640, 640))
    
    try:
    
        response = await detectionService.execute(frame_cropped)
        filtered_detections = [
            detection for detection in response['results']
            if detection['score'] >= 0.3
        ]
    
        frame_cropped = use_case.execute(
            frame=frame_cropped,
            place=config['place'],
            detections=filtered_detections
        )
    
    except Exception as e:
        print(f"Error en la detección: {e}")
        
    cv2.imwrite('output.jpg', cv2.resize(frame_cropped, (640, 640)))

async def main():
    await streamCaptureService.start_stream(lambda frame: callback(frame, config))

if __name__ == '__main__':
    asyncio.run(main())
