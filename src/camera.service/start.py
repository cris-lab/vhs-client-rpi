# main_app.py (o como se llame tu archivo principal)

import numpy as np
import cv2
import argparse
import os
import sys
import asyncio
import multiprocessing as mp
from src.utils import crop_and_resize_roi_padded
from types import SimpleNamespace
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from src import StreamCapture as vs
from dotenv import load_dotenv
# from src.OpenAiService import OpenAiService # No usada aquí, mantenemos comentario
from src.process_frame import process_frame, cleanup_tracks
from src.config_utils import load_config
import traceback
import aiohttp
from datetime import datetime # Nueva importación para timestamps
import time # Nueva importación para time.sleep

# Parche temporal para compatibilidad con numpy >=1.24
if not hasattr(np, 'float'):
    np.float = float

load_dotenv()

# Argumentos
parser      = argparse.ArgumentParser(description='VHS Camera Stream Service')
parser.add_argument('--stream-id', type=str, required=True, help='ID del stream a procesar')

args        = parser.parse_args()
stream_id   = args.stream_id

FILE_PATH = '/var/lib/vhs'
config_path = os.path.join(FILE_PATH, 'config.json')
config = load_config(config_path)

stream = next((s for s in config.get('streams', []) if s['id'] == stream_id), None)

if stream is None:
    raise ValueError(f"No se encontró la transmisión con ID {stream_id}")

class mainStreamClass:
    
    def __init__(self):
        self.camProcess         = None
        self.cam_queue          = None
        self.stopbit            = None
        self.camlink            = stream['input']['url']
        self.framerate          = stream['input']['fps']
        self.session            = None
        self.exit_code          = 0 # 0 para salida limpia, 1 para error
        self.tracker            = BYTETracker(
            SimpleNamespace(
                track_thresh    = stream['tracker'].get('track_thresh', 0.25),
                track_buffer    = stream['tracker'].get('track_buffer', 30),
                match_thresh    = stream['tracker'].get('match_thresh', 0.8),
                mot20           = stream['tracker'].get('mot20', False),
            ), 
            frame_rate          =stream['input']['fps']
        )

    async def startMain(self):
        self.cam_queue = mp.Queue(maxsize=20) 
        self.stopbit = mp.Event()
        self.camProcess = vs.StreamCapture(
            self.camlink,
            self.stopbit,
            self.cam_queue,
            self.framerate
        )
        self.camProcess.start()

        await asyncio.sleep(0.5)
        async with aiohttp.ClientSession() as session:
            self.session = session

            try:
                while True:
                    if not self.camProcess.is_alive():
                        print(f"[{datetime.now()}] [mainStreamClass] camProcess terminó. Saliendo.")
                        break

                    latest_frame = None

                    # Vaciamos la cola, procesaremos solo el último frame disponible
                    while not self.cam_queue.empty():
                        cmd, frame = self.cam_queue.get_nowait()
                        
                        if cmd == vs.StreamCommands.ERROR:
                            print(f"[{datetime.now()}] [mainStreamClass] StreamCapture reportó error. Esperando fin del proceso.")
                            latest_frame = None
                            break  # No procesamos más si se reporta error

                        if cmd == vs.StreamCommands.FRAME and frame is not None:
                            latest_frame = frame  # Nos quedamos con el último frame

                    # Procesar solo si hay un frame nuevo
                    if latest_frame is not None:
                        detections_found = False
                        
                        if not detections_found:
                            cv2.putText(frame, "Buscando...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        if os.getenv("WINDOW_ENABLED") == "1":
                            cv2.imshow('Cam: ' + self.camlink, frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.exit_code = 0
                                break

                        if os.getenv("SEND_TO_BACKEND") == "1":
                            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                            if is_success:
                                image_bytes = im_buf_arr.tobytes()
                                await self._send_to_backend(image_bytes)
                    
                    await asyncio.sleep(0.005)

            except Exception as e:
                print(f"[{datetime.now()}] [mainStreamClass] !!! ERROR en el bucle principal: {e}")
                print(traceback.format_exc())
                self.exit_code = 1

            finally:
                self.stopCamStream()
                cv2.destroyAllWindows() 
                sys.exit(self.exit_code)

    
    async def _send_to_backend(self, image_bytes: bytes):
        backend_url = f"http://127.0.0.1:8000/processed_stream/{stream_id}"
        try:
            async with self.session.post(backend_url, data=image_bytes, timeout=10) as resp:
                resp.raise_for_status()
                await resp.read()
        except aiohttp.ClientError as e:
            print(f"[{datetime.now()}] [mainStreamClass] Error al enviar frame procesado para {stream_id} (ClientError): {e}")
        except asyncio.TimeoutError:
            print(f"[{datetime.now()}] [mainStreamClass] Tiempo de espera agotado al enviar frame procesado para {stream_id}.")
        except Exception as e:
            print(f"[{datetime.now()}] [mainStreamClass] Error general al enviar frame procesado para {stream_id}: {e}")

    def stopCamStream(self):
        print(f"[{datetime.now()}] [mainStreamClass] En stopCamStream")
        if self.stopbit is not None:
            self.stopbit.set() # Señaliza al subproceso de la cámara que se detenga
            try:
                while not self.cam_queue.empty():
                    try:
                        _ = self.cam_queue.get_nowait()
                    except:
                        break
                self.cam_queue.close()
                self.cam_queue.join_thread()
                
                # Dale un tiempo para que el proceso de la cámara termine de forma limpia
                # Si StreamCapture se cerró limpiamente (exit_code 0), join debería ser rápido.
                # Si se cerró con error (exit_code 1), ya estará muerto.
                self.camProcess.join(timeout=5) 
                if self.camProcess.is_alive():
                    print(f"[{datetime.now()}] [mainStreamClass] Advertencia: El proceso de la cámara no se cerró a tiempo. Terminando forzosamente.")
                    self.camProcess.terminate()
                    self.camProcess.join()
            except Exception as e:
                print(f"[{datetime.now()}] [mainStreamClass] Error durante la limpieza de la cámara/cola: {e}")

        print(f"[{datetime.now()}] [mainStreamClass] Camera stream stopped")
        print(f"[{datetime.now()}] [mainStreamClass] Exiting mainStreamClass")

if __name__ == "__main__":
    # El bucle de reintento de "último recurso" queda aquí
    RETRY_INTERVAL_SECONDS_SYSTEMD_BASED = 60 # Espera un minuto antes de que systemd intente de nuevo
                                              # Esto debería ser el RestartSec de systemd si quieres que él lo haga.
                                              # Si Python no debe reintentar, este bucle NO debería estar aquí.
    
    # Para tu requisito, el bucle principal de reintento ya no es necesario aquí.
    # El proceso de StreamCapture lo maneja internamente.
    # mainStreamClass solo se ejecuta una vez y termina si StreamCapture falla persistentemente.
    # Systemd será el encargado de reiniciar todo el mainStreamClass si este se cae.
    
    mc = mainStreamClass()
    asyncio.run(mc.startMain())