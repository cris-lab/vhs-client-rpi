from src.StreamCaptureService import StreamCaptureService
from src.DetectionService import DetectionService
from src.Tracker import Tracker
#from sort import Sort
import asyncio
import cv2
import numpy as np
import time

# Inicialización de servicios
streamCaptureService = StreamCaptureService()
detectionService = DetectionService()
tracker = Tracker()

config = {'roi': [(260, 160), (1300, 1200)]}  # Tupla en lugar de lista para mayor eficiencia

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
    
    response = await detectionService.execute(frame_cropped)
    filtered_detections = [
        detection for detection in response['results']
        if detection['score'] >= 0.3
    ]
    
    for detection in filtered_detections:
        print(detection)
        x1, y1, x2, y2 = map(int, detection['bbox'])
        cv2.rectangle(frame_cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite('output.jpg', cv2.resize(frame_cropped, (640, 640)))


async def main():
    await streamCaptureService.start_stream(lambda frame: callback(frame, config))

if __name__ == '__main__':
    asyncio.run(main())
