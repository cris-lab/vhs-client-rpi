import degirum_tools
import threading
import os
import json
import time 
import numpy as np
import cv2
import uuid as uuid_lib
import traceback
from typing import Dict, Any, List, Tuple, Optional 
from src.ModelLoader import ModelLoader

class EventProcessor:

    def __init__(self, config: Dict[str, Any], stream: Dict[str, Any]):
        print("[DEBUG_INIT] Inicializando EventProcessor...")
        self.config = config
        self.stream = stream
        self.callbacks = []
        self.event_tracker: Dict[int, Dict[str, Any]] = {} 
        self.lock = threading.Lock()
        self.base_storage_dir = "/opt/vhs/storage/detecciones"
        os.makedirs(self.base_storage_dir, exist_ok=True)
        print(f"[DEBUG_INIT] Directorio de almacenamiento de detecciones: {self.base_storage_dir}")

        self.MAX_FACE_INFERENCES_PER_PERSON = 3
        self.INFERENCE_TIMEOUT_SECONDS = 15
        self.CLEANUP_GRACE_PERIOD = 1.0
        self.MIN_FACE_CROP_DIMENSION = 30 # 🟢 Nuevo: Dimensión mínima para un recorte de rostro válido

        print("[DEBUG_INIT] Cargando modelos de rostro para CombiningCompoundModel...")
        try:
            self.face_feature_model = degirum_tools.CombiningCompoundModel(
                ModelLoader('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1').load_model(),
                ModelLoader('yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1').load_model()
            )
            print("[DEBUG_INIT] Modelos de rostro cargados exitosamente para EventProcessor.")
        except Exception as e:
            print(f"[ERROR_INIT] Fallo al cargar los modelos de rostro: {e}")
            raise

    def add_callback(self, callback: callable):
        print("[INFO] Callback agregado.")
        self.callbacks.append(callback)

    def on_cross(self, event: Any):
        print(f"[DEBUG_ON_CROSS_RAW] on_cross llamado. Tipo de evento: {type(event)}, Contenido: {event}")

        if not isinstance(event, dict):
            print(f"[ERROR_ON_CROSS] Evento de entrada inválido: {event}. Se esperaba un diccionario. Ignorando.")
            return

        print(f"[DEBUG_ON_CROSS] on_cross llamado para TID: {event.get('tid')}")
        event['place_code'] = self.config.get('code')
        event['stream_code'] = self.stream.get('code')
        print(f"[EVENT] 🚶 Persona cruzó la línea: {event.get('tid')}")

    def on_cross_inference_gender_age(self, event: Any):
        print(f"[DEBUG_ON_CROSS_INF_RAW] on_cross_inference_gender_age llamado. Tipo de evento: {type(event)}, Contenido: {event}")
        
        if not isinstance(event, dict):
            print(f"[ERROR_ON_CROSS_INF] TypeError: 'int' object is not subscriptable. Se recibió un entero ({event}) en lugar de un diccionario. EL CÓDIGO QUE LLAMA DEBE PASAR EL OBJETO 'EVENT' COMPLETO, NO SOLO EL TID.")
            return

        if not isinstance(event.get('tid'), int):
            print(f"[ERROR_ON_CROSS_INF] TID inválido. Se esperaba un 'tid' entero, pero se recibió: {event.get('tid')} de tipo {type(event.get('tid'))}. Ignorando.")
            return

        print(f"[DEBUG_ON_CROSS_INF] on_cross_inference_gender_age llamado para TID: {event.get('tid')}")
        event['place_code'] = self.config.get('code')
        event['stream_code'] = self.stream.get('code')
        event['features'] = []
        event['uuid'] = event.get('uuid') or str(uuid_lib.uuid4())
        event['start_time'] = time.time()
        event['is_complete'] = False
        # 🟢 Nuevos campos para manejo de errores y reintentos
        event['inference_in_progress'] = False
        event['inference_failures'] = 0
        event['last_inference_time'] = 0
        event['error_during_inference'] = None

        print(f"[DEBUG_ON_CROSS_INF] Evento preparado: {event}")

        with self.lock:
            if not isinstance(event['tid'], int):
                print(f"[ERROR_ON_CROSS_INF] TID inválido antes de añadir al tracker: {event['tid']}. Saltando.")
                return
            self.event_tracker[event['tid']] = event
            print(f"[DEBUG_ON_CROSS_INF] Evento {event['tid']} añadido al tracker. Tracker actual: {list(self.event_tracker.keys())}")

    def save_person_data_to_json(self, person_data: Dict[str, Any]) -> bool:
        print(f"[DEBUG_SAVE_JSON] save_person_data_to_json llamado para UUID: {person_data.get('uuid')}")
        uuid_val = person_data.get("uuid")
        if not uuid_val:
            print("[❌_SAVE_JSON] UUID faltante en los datos de la persona, no se guarda.")
            return False

        filepath = os.path.join(self.base_storage_dir, f"{uuid_val}.json")
        print(f"[DEBUG_SAVE_JSON] Intentando escribir JSON en: {filepath}")
        try:
            with open(filepath, 'w') as f:
                json.dump(person_data, f, indent=4)
            print(f"[✔_SAVE_JSON] Guardado JSON exitoso: {filepath}")
            return True
        except Exception as e:
            print(f"[❌_SAVE_JSON] Error al guardar JSON '{filepath}': {e}")
            return False
        
    def analyze(self, result, frame):
        print("[DEBUG_ANALYZE] analyze llamado.")
        MIN_FACE_SCORE = 0.6
        MAX_FACE_DISTANCE = 100

        face_detecciones = [
            d for d in result.results
            if d.get('label', '').lower() == 'human face' and d.get('score', 0.0) >= MIN_FACE_SCORE
        ]
        print(f"[DEBUG_ANALYZE] Total rostros filtrados: {len(face_detecciones)}")
        
        current_tids = {res.get('track_id') for res in result.results if res.get('track_id') is not None}

        with self.lock:
            event_tracker_tids = set(self.event_tracker.keys())
            print(f"[DEBUG_ANALYZE] TIDs activos en el tracker: {event_tracker_tids}")
            
            tracks_to_delete = [
                tid for tid in event_tracker_tids
                if tid not in current_tids
            ]
            print(f"[DEBUG_ANALYZE] TIDs a eliminar: {tracks_to_delete}")

        valid_results = [res for res in result.results if isinstance(res, dict) and isinstance(res.get('track_id'), int)]

        for res in valid_results:
            track_id = res.get('track_id')
            
            if track_id not in event_tracker_tids:
                continue

            bbox = res.get('bbox')
            if not bbox:
                continue

            print(f"[DEBUG_ANALYZE] Procesando TID {track_id} con bbox {bbox}")
            center = self._get_center(bbox)

            with self.lock:
                enriched_event = self.event_tracker.get(track_id)
            
            if enriched_event is None or enriched_event.get("is_complete", False):
                continue

            if enriched_event.get("direction", "").lower() == "down":
                print(f"[🛑] TID {track_id} saltado por dirección 'down'")
                continue

            faces_inside = []
            bx1, by1, bx2, by2 = bbox
            box_top = by1
            box_bottom = by2
            box_height = box_bottom - box_top
            zone2_limit = box_top + 2 * box_height / 3
            
            print(f"[DEBUG_ANALYZE] Bbox de persona: {bbox}. Límite de zona 2: {zone2_limit}")

            for f in face_detecciones:
                face_bbox = f.get('bbox')
                if not face_bbox:
                    continue
                fx1, fy1, fx2, fy2 = face_bbox
                face_center_x, face_center_y = self._get_center(face_bbox)

                if (
                    self._is_inside(face_bbox, bbox) and
                    face_center_y <= zone2_limit and
                    self._distance((face_center_x, face_center_y), center) < MAX_FACE_DISTANCE
                ):
                    faces_inside.append(f)

            print(f"[DEBUG_ANALYZE] Rostros válidos dentro del bbox: {len(faces_inside)}")

            if not faces_inside:
                continue

            best_face = max(faces_inside, key=lambda f: self._face_score(f['bbox'], center))
            x1, y1, x2, y2 = map(int, best_face['bbox'])
            face_crop = frame[y1:y2, x1:x2]
            
            # 🟢 Verificación de recorte de rostro más robusta
            if face_crop.shape[0] < self.MIN_FACE_CROP_DIMENSION or face_crop.shape[1] < self.MIN_FACE_CROP_DIMENSION:
                print(f"[ADVERTENCIA_ANALYZE] Recorte de rostro para TID {track_id} es demasiado pequeño: {face_crop.shape}. Se salta la inferencia.")
                continue

            print(f"[INFO_ANALYZE] Recorte de rostro para TID {track_id} tiene forma: {face_crop.shape}")

            with self.lock:
                enriched_event_current_state = self.event_tracker.get(track_id)
                if (enriched_event_current_state and 
                    not enriched_event_current_state.get('inference_in_progress', False) and 
                    len(enriched_event_current_state['features']) < self.MAX_FACE_INFERENCES_PER_PERSON and
                    enriched_event_current_state.get('inference_failures', 0) < 3): # 🟢 Limitar los reintentos

                    enriched_event_current_state['inference_in_progress'] = True
                    enriched_event_current_state['last_inference_time'] = time.time()
                    enriched_event_current_state['inference_start_time'] = time.time() 
                    print(f"[INFO] 🔍 Iniciando inferencia para TID {track_id}")
                    self.process_face_inference_async(enriched_event_current_state, face_crop)
                else:
                    print(f"[DEBUG_ANALYZE] TID {track_id} ya alcanzó el máximo de inferencias, hay una en curso, o ha fallado demasiadas veces. No se procesa.")
        
        print(f"[DEBUG_ANALYZE] Finalizado el bucle. Llamando a clear_event_tracker con tids: {tracks_to_delete}")
        self.clear_event_tracker(tracks_to_delete)
        
    def clear_event_tracker(self, stale_tids: set):
        print(f"[DEBUG_CLEAR_TRACKER] clear_event_tracker llamado con stale_tids: {stale_tids}")
        with self.lock:
            print("[DEBUG_CLEAR_TRACKER] Adquiriendo lock para clear_event_tracker.")
            tids_to_remove_from_tracker = set()
            
            stale_tids_filtered = [tid for tid in stale_tids if tid is not None]
            
            for tid in stale_tids_filtered:
                print(f"[DEBUG_CLEAR_TRACKER] Evaluando TID {tid} para limpieza.")
                event_data = self.event_tracker.get(tid)
                
                if event_data is None or event_data.get('is_complete', False):
                    continue

                # 🟢 Solo se considera la limpieza si el evento ha existido durante el período de gracia
                if time.time() - event_data.get('start_time', 0) < self.CLEANUP_GRACE_PERIOD:
                    print(f"[DEBUG_CLEAR_TRACKER] TID {tid} es muy nuevo (dentro del período de gracia). Se salta la limpieza.")
                    continue
                
                print(f"[DEBUG_CLEAR_TRACKER] TID {tid} encontrado en el tracker. Estado de inferencia: in_progress={event_data.get('inference_in_progress')}, failures={event_data.get('inference_failures')}")
                should_save_and_remove = False
                
                if not event_data.get('inference_in_progress', False):
                    print(f"[DEBUG_CLEAR_TRACKER] TID {tid}: No hay inferencia en progreso. Marcado para guardar y eliminar.")
                    should_save_and_remove = True
                elif time.time() - event_data.get('inference_start_time', 0) > self.INFERENCE_TIMEOUT_SECONDS:
                    print(f"[ALERTA_CLEAR_TRACKER] Inferencia para TID {tid} excedió el tiempo límite ({self.INFERENCE_TIMEOUT_SECONDS}s). Marcado para guardar estado actual y eliminar.")
                    event_data['error_during_inference'] = "Inference timed out."
                    should_save_and_remove = True
                elif event_data.get('inference_failures', 0) >= 2 and not event_data.get('inference_in_progress', False):
                    print(f"[INFO_CLEAR_TRACKER] TID {tid}: Ha tenido múltiples fallos de inferencia. Forzando guardado y eliminación.")
                    should_save_and_remove = True
                else:
                    print(f"[DEBUG_CLEAR_TRACKER] TID {tid}: Inferencia en progreso y dentro del tiempo límite, o esperando reintento. No se guarda ni elimina aún.")

                if should_save_and_remove:
                    print(f"[DEBUG_CLEAR_TRACKER] Intentando guardar JSON final para TID {tid}.")
                    if self.save_person_data_to_json(event_data):
                        event_data['is_complete'] = True
                        print(f"[EVENT] 🗑️ Removido y guardado TID {tid} al finalizar seguimiento o por timeout/fallo.")
                        tids_to_remove_from_tracker.add(tid)
                    else:
                        print(f"[ADVERTENCIA_CLEAR_TRACKER] No se pudo guardar JSON para TID {tid}. Se mantiene en tracker para reintentar o depurar.")
            
            for tid in tids_to_remove_from_tracker:
                if tid in self.event_tracker:
                    del self.event_tracker[tid]
                    print(f"[DEBUG_CLEAR_TRACKER] TID {tid} eliminado del event_tracker. Nuevo tamaño: {len(self.event_tracker)}")
            print("[DEBUG_CLEAR_TRACKER] Liberando lock después de clear_event_tracker.")
        print("[DEBUG_CLEAR_TRACKER] Fin de clear_event_tracker.")

    def process_face_inference_async(self, enriched_event: Dict[str, Any], face_crop: np.ndarray):
        print(f"[DEBUG_ASYNC_CALL] Iniciando hilo asíncrono para inferencia de rostro para TID {enriched_event.get('tid')}")
        
        # 🟢 Validar de forma robusta antes de iniciar el hilo
        if not isinstance(face_crop, np.ndarray) or face_crop.size == 0 or face_crop.shape[0] < self.MIN_FACE_CROP_DIMENSION or face_crop.shape[1] < self.MIN_FACE_CROP_DIMENSION:
            print(f"[ERROR_ASYNC_CALL] face_crop inválido o muy pequeño para TID {enriched_event.get('tid')}. No se inicia hilo de inferencia.")
            with self.lock:
                if enriched_event.get('tid') in self.event_tracker:
                    self.event_tracker[enriched_event.get('tid')]['inference_in_progress'] = False
                    self.event_tracker[enriched_event.get('tid')]['error_during_inference'] = "Invalid or too small face_crop provided to async thread."
                    self.event_tracker[enriched_event.get('tid')]['inference_failures'] = self.event_tracker[enriched_event.get('tid')].get('inference_failures', 0) + 1
                    print(f"[DEBUG_ASYNC_CALL] Fallo de face_crop para TID {enriched_event.get('tid')}. Fallos acumulados: {self.event_tracker[enriched_event.get('tid')]['inference_failures']}")
            return

        threading.Thread(
            target=self._process_face_inference,
            args=(enriched_event, face_crop),
            daemon=True
        ).start()
        print(f"[DEBUG_ASYNC_CALL] Hilo para TID {enriched_event.get('tid')} iniciado.")

    def _process_face_inference(self, enriched_event, face_crop):
        tid = enriched_event.get('tid', 'unknown')
        uuid = enriched_event.get('uuid', 'unknown')
        print(f"[DEBUG_INFERENCE] Hilo async iniciado para TID {tid}, UUID {uuid}")
        
        try:
            print(f"[DEBUG_INFERENCE_TRY] Realizando inferencia para TID {tid} con recorte de forma: {face_crop.shape}...")
            inference = self.face_feature_model(face_crop)
            print(f"[DEBUG_INFERENCE_TRY] Inferencia completada para TID {tid}. Resultados: {inference.results}")

            with self.lock:
                if enriched_event.get('tid') not in self.event_tracker:
                     print(f"[ADVERTENCIA_INFERENCE] El evento para TID {tid} ya fue eliminado del tracker. Se descartan los resultados de la inferencia.")
                     return

                enriched_event['features'].append(inference.results)
                enriched_event['inference_in_progress'] = False
                enriched_event['last_inference_time'] = time.time()
                print(f"[✔] Inferencia completada y estado actualizado para TID {tid}. Total features: {len(enriched_event['features'])}")

                if len(enriched_event['features']) >= self.MAX_FACE_INFERENCES_PER_PERSON:
                    print(f"[INFO] 🎉 TID {tid} alcanzó el máximo de inferencias. Intentando guardar y finalizar.")
                    if self.save_person_data_to_json(enriched_event):
                        enriched_event['is_complete'] = True 
                        print(f"[✔] JSON guardado completo para UUID {uuid}")
                        
        except Exception as e:
            print(f"[❌] Error inferencia rostro para TID {tid} UUID {uuid}.")
            traceback.print_exc()
            with self.lock:
                if enriched_event.get('tid') in self.event_tracker:
                    self.event_tracker[enriched_event.get('tid')]['inference_in_progress'] = False
                    self.event_tracker[enriched_event.get('tid')]['error_during_inference'] = f"Inference failed with error: {str(e)}"
                    self.event_tracker[enriched_event.get('tid')]['inference_failures'] = self.event_tracker[enriched_event.get('tid')].get('inference_failures', 0) + 1
                    print(f"[DEBUG_INFERENCE] Fallo de inferencia para TID {tid}. Fallos acumulados: {self.event_tracker[enriched_event.get('tid')]['inference_failures']}")

    def _get_center(self, bbox: Optional[Tuple[float, float, float, float]]) -> Tuple[float, float]:
        if bbox is None:
            return (0.0, 0.0) 
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_area(self, bbox: Optional[Tuple[float, float, float, float]]) -> float:
        if bbox is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1))

    def _face_score(self, bbox_face: Optional[Tuple[float, float, float, float]], person_center: Tuple[float, float]) -> float:
        if bbox_face is None:
            return -1.0 
        face_center = self._get_center(bbox_face)
        score = self._bbox_area(bbox_face) / (1 + self._distance(face_center, person_center))
        return score

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
        return dist

    def _is_inside(self, small_box: Optional[Tuple[float, float, float, float]], big_box: Optional[Tuple[float, float, float, float]]) -> bool:
        if small_box is None or big_box is None:
            return False
        sx1, sy1, sx2, sy2 = small_box
        bx1, by1, bx2, by2 = big_box
        is_inside = sx1 >= bx1 and sy1 >= by1 and sx2 <= bx2 and sy2 <= by2
        return is_inside