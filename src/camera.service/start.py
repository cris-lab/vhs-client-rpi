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
from src.InOutUseCase import InOutUseCase

import traceback
import aiohttp # <<<< IMPORTANTE: Añadir para peticiones HTTP asíncronas

# Parche temporal para compatibilidad con numpy >=1.24
if not hasattr(np, 'float'): # <<<< ESTA ES LA LÍNEA CORRECTA
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

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"ERROR: El archivo de configuración no se encontró en '{config_path}'.")
    print("Por favor, asegúrese de que 'config.json' exista en la ruta especificada.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"ERROR: El archivo '{config_path}' no es un JSON válido.")
    print("Por favor, revise la sintaxis del archivo de configuración.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Ocurrió un error inesperado al cargar la configuración: {e}")
    sys.exit(1)

stream = next((s for s in config.get('streams', []) if s['id'] == stream_id), None)

if stream is None:
    raise ValueError(f"No se encontró la transmisión con ID {stream_id}")

new_width               = 800
new_height              = 450

detection_service       = DetectionService(url='http://127.0.0.1:5000/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
gender_service          = DetectionService(url='http://127.0.0.1:5001/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
age_service             = DetectionService(url='http://127.0.0.1:5002/inference', time_zone=config.get('time_zone'), schedule=config.get('detection_schedule'))
# use_case                = InOutUseCase(
#     stream=stream,
#     counter_interpolation=stream['counter_interpolation'],
#     cross_line=stream['detection_cross_line'],
#     centroid_orientation=stream['centroid_orientation']
# )

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
        # >>>>>> CAMBIO CLAVE: Inicializar sesión aiohttp aquí, será None hasta startMain
        self.session = None
        self.exit_code = 0

    async def startMain(self):
        self.cam_queue = mp.Queue(maxsize=100)
        self.stopbit = mp.Event()
        self.camProcess = vs.StreamCapture(
            self.camlink,
            self.stopbit,
            self.cam_queue,
            self.framerate
        )
        self.camProcess.start()

        await asyncio.sleep(0.5)

        # >>>>>> CAMBIO CLAVE: Crear la sesión aiohttp una vez al inicio
        # Usamos un 'async with' para asegurar que la sesión se cierre limpiamente
        async with aiohttp.ClientSession() as session:
            self.session = session # Guardamos la sesión para usarla en self._send_to_backend

            try:
                while True:
                    
                    print('v')
                    
                    if not self.camProcess.is_alive():
                        print("[Fatal] camProcess murió inesperadamente sin reportar error.")
                        self.exit_code = 1
                        break
                    
                    if not self.cam_queue.empty():
                        cmd, frame = self.cam_queue.get()
                        
                        if cmd == vs.StreamCommands.ERROR:
                            print("StreamCapture reportó error fatal.")
                            raise ValueError("StreamCapture reportó error fatal.")

                        if cmd == vs.StreamCommands.FRAME and frame is not None:

                            frame = cv2.resize(frame, (new_width, new_height))

                            print('Processing frame from camera stream')
                            response = await detection_service.execute(frame)
                            detections_found = False

                            print(response)

                            # >>>>>> Lógica de procesamiento y dibujo de bounding boxes
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
                                    results = self.tracker.update(output_results, img_info, img_size)

                                    for track in results:
                                        x1, y1, w, h = track.tlwh
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
                                        track_id = track.track_id

                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                        if track_id not in self.person_data:
                                            head_h = int(h * 0.3)
                                            head_img = frame[y1:min(y1+head_h, frame.shape[0]), x1:min(x2, frame.shape[1])]

                                            gender  = "U"
                                            age     = 0

                                            # gender_prediction_response = await gender_service.execute(head_img)
                                            # if gender_prediction_response.get('status', False) and gender_prediction_response.get('results'):
                                            #     if gender_prediction_response['results']:
                                            #         gender = gender_prediction_response['results'][0].get('gender', "U")
                                            # else:
                                            #     print(f"No se pudo obtener el género para track {track_id}: {gender_prediction_response.get('message', 'Error desconocido')}")

                                            self.person_data[track_id] = {
                                                "gender": gender,
                                                "age": age,
                                                "head_bbox": [x1, y1, x2 - x1, head_h],
                                                "timestamp": time.time()
                                            }

                                        gender = self.person_data[track_id]["gender"]
                                        age = self.person_data[track_id]["age"]

                                        cv2.putText(frame, f'ID:{track_id} {gender} {age}',
                                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                    (0, 255, 255), 2)

                                else:
                                    print("No detections to track from the service's results.")
                            else:
                                print(f"No results from detection service, reason: {response.get('message', 'Unknown reason')}")

                            if not detections_found:
                                cv2.putText(frame, "Buscando...", (20, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                            
                            # Convertir frame OpenCV a bytes para el envío
                            # Puedes elegir el formato (JPEG es común y eficiente para streaming)
                            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                            if is_success:
                                image_bytes = im_buf_arr.tobytes()
                                await self._send_to_backend(image_bytes)
                            else:
                                print("Error al codificar frame para envío al backend.")

                            if os.getenv("WINDOW_ENABLED") == "1":
                                frame_display = cv2.resize(frame, (new_width, new_height))
                                cv2.imshow('Cam: ' + self.camlink, frame_display)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                
                    else:
                        await asyncio.sleep(0.001)

            except KeyboardInterrupt:
                print('Caught Keyboard interrupt. Shutting down.')
                self.exit_code = 0
            except Exception as e:
                print('Caught Main Exception:')
                print(f"Error: {e}")
                print(traceback.format_exc())
                self.exit_code = 1
            finally:
                self.stopCamStream()
                cv2.destroyAllWindows()
                sys.exit(self.exit_code)

    # Enviar frame procesado al backend
    async def _send_to_backend(self, image_bytes: bytes):
        backend_url = f"http://127.0.0.1:8000/processed_stream/{stream_id}"
        try:
            # Reutiliza la sesión 'self.session' creada en startMain
            async with self.session.post(backend_url, data=image_bytes, timeout=10) as resp: # Añadir timeout
                resp.raise_for_status() # Lanza una excepción para códigos de estado HTTP 4xx/5xx
                await resp.read()  # Asegura que la conexión se libere y los datos se consuman
        except aiohttp.ClientError as e:
            print(f"Error al enviar frame procesado para {stream_id} (ClientError): {e}")
        except asyncio.TimeoutError: # Captura si el timeout de aiohttp se cumple
            print(f"Tiempo de espera agotado al enviar frame procesado para {stream_id}.")
        except Exception as e:
            print(f"Error general al enviar frame procesado para {stream_id}: {e}")

    def stopCamStream(self):
        print('in stopCamStream')
        if self.stopbit is not None:
            self.stopbit.set()
            try:
                while not self.cam_queue.empty():
                    try:
                        _ = self.cam_queue.get(block=False)
                    except Exception:
                        break
                self.cam_queue.close()
                self.cam_queue.join_thread()
                self.camProcess.join(timeout=5)
                if self.camProcess.is_alive():
                    print("Advertencia: El proceso de la cámara no se cerró a tiempo. Terminando forzosamente.")
                    self.camProcess.terminate()
                    self.camProcess.join()
            except Exception as e:
                print(f"Error durante la limpieza de la cámara/cola: {e}")

        print('Camera stream stopped')
        print('Exiting mainStreamClass')

if __name__ == "__main__":
    mc = mainStreamClass()
    asyncio.run(mc.startMain())