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
import multiprocessing as mp
import StreamCapture as vs
from dotenv import load_dotenv
from types import SimpleNamespace
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from src.DetectionService import DetectionService
from src.Tracker import Tracker
from src.InOutUseCase import InOutUseCase

# Parche temporal para compatibilidad con numpy >=1.24
if not hasattr(np, 'float'):
    np.float = float

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

new_width               = 800
new_height              = 450

detection_service       = DetectionService(url='http://127.0.0.1:5000/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
gender_service          = DetectionService(url='http://127.0.0.1:5001/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
age_service             = DetectionService(url='http://127.0.0.1:5002/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
use_case                = InOutUseCase(
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

class mainStreamClass:
    def __init__(self):
        self.camProcess = None
        self.cam_queue  = None
        self.stopbit    = None
        self.camlink    = stream['input']['url']
        self.framerate  = stream['input']['fps']

        self.tracker = BYTETracker(SimpleNamespace(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False
        ), frame_rate=self.framerate)
        
        self.person_data = {}

    def startMain(self):
        self.cam_queue = mp.Queue(maxsize=100)
        time.sleep(3)

        self.stopbit = mp.Event()
        self.camProcess = vs.StreamCapture(
            self.camlink,
            self.stopbit,
            self.cam_queue,
            self.framerate
        )
        self.camProcess.start()

        lastFTime = time.time()

        try:
            while True:
                if not self.cam_queue.empty():
                    cmd, frame = self.cam_queue.get()
                    lastFTime = time.time()

                    if cmd == vs.StreamCommands.FRAME and frame is not None:
                        
                        frame = cv2.resize(frame, (new_width, new_height))
                        
                        print('Processing frame from camera stream')
                        response = asyncio.run(detection_service.execute(frame))
                        detections_found = False

                        if response and 'status' in response and not response['status']:
                            # Si el servicio de deteccion no esta disponible, se deberia intentar reinicialo
                            # nombre del servicio systemd vhs.detection.service
                            print("Detection service is not available, trying to restart it")
                            try:
                                subprocess.run(['systemctl', 'restart', 'vhs.detection.service'], check=True)
                                print("Detection service restarted successfully")
                                response = asyncio.run(detection_service.execute(frame))
                            except subprocess.CalledProcessError as e:
                                print(f"Failed to restart detection service: {e}")
                         
                        print(response)   

                        if response and 'results' in response:
                            output_results = np.array([
                                det['bbox'] + [det['score']]
                                for det in response['results']
                                if 'bbox' in det and 'score' in det
                            ])

                            if output_results.shape[0] > 0:
                                detections_found = True
                                img_info = frame.shape[:2]
                                img_size = (frame.shape[0], frame.shape[1])
                                print(f'Image info: {img_info}, Image size: {img_size}')
                                results = self.tracker.update(output_results, img_info, img_size)
                                print(f'Tracking results: {results}')
                                
                                # Procesar cada track
                                for track in results:
                                    x1, y1, w, h = track.tlwh
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
                                    track_id = track.track_id

                                    # Dibujar caja
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                    if track_id not in self.person_data:
                                        # Extraer "cabeza" como 30% superior del bbox
                                        head_h = int(h * 0.3)
                                        head_img = frame[y1:y1+head_h, x1:x2]
                                        
                                        # Aquí puedes meter llamadas a tus modelos reales
                                        gender  = "U"  # o predict_gender(head_img)
                                        age     = 0      # o predict_age(head_img)

                                        gender_prediction_results   = asyncio.run(gender_service.execute(frame))
                                        #age_prediction_results      = asyncio.run(age_service.execute(frame))
                                        
                                        print(gender_prediction_results)
                                        
                                        

                                        self.person_data[track_id] = {
                                            "gender": gender,
                                            "age": age,
                                            "head_bbox": [x1, y1, x2 - x1, head_h],
                                            "timestamp": time.time()
                                        }

                                    # Recuperar género y edad
                                    gender = self.person_data[track_id]["gender"]
                                    age = self.person_data[track_id]["age"]

                                    # Mostrar texto en el frame
                                    cv2.putText(frame, f'ID:{track_id} {gender} {age}',
                                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 255, 255), 2)

                                
                                # Aquí puedes pintar rectángulos si quieres
                            else:
                                print("No detections to track")
                        else:
                            print("No results from detection service")

                        # Mostrar "Buscando..." si no hay detecciones
                        if not detections_found:
                            cv2.putText(frame, "Buscando...", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        if os.getenv("WINDOW_ENABLED") == "1":
                            frame = cv2.resize(frame, (new_width, new_height))
                            cv2.imshow('Cam: ' + self.camlink, frame)
                            cv2.waitKey(1)

        except KeyboardInterrupt:
            print('Caught Keyboard interrupt')

        except:
            e = sys.exc_info()
            print('Caught Main Exception')
            print(e)

        self.stopCamStream()
        cv2.destroyAllWindows()

    def stopCamStream(self):
        print('in stopCamStream')
        if self.stopbit is not None:
            self.stopbit.set()
            while not self.cam_queue.empty():
                try:
                    _ = self.cam_queue.get()
                except:
                    break
            self.cam_queue.close()
            self.camProcess.join()
            
        print('Camera stream stopped')
        print('Exiting mainStreamClass')
        sys.exit(9)

if __name__ == "__main__":
    mc = mainStreamClass()
    mc.startMain()
