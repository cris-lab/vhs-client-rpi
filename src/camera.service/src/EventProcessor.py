import degirum_tools
import threading
import os
import json
import time # Importar time para la marca de tiempo del timeout
from src.ModelLoader import ModelLoader
from typing import Dict, Any, List

class EventProcessor:

    def __init__(self, config: Dict[str, Any], stream: Dict[str, Any]):
        self.config = config
        self.stream = stream
        self.callbacks = []
        # Cambiamos event_tracker a un diccionario para un acceso más eficiente por track_id
        self.event_tracker: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.base_storage_dir = "/opt/vhs/storage/detections"
        os.makedirs(self.base_storage_dir, exist_ok=True)

        self.MAX_FACE_INFERENCES_PER_PERSON = 3
        # Añadimos un tiempo máximo para la inferencia de un rostro, por si se "cuelga"
        self.INFERENCE_TIMEOUT_SECONDS = 30 # Por ejemplo, 30 segundos
        
        self.face_feature_model = degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1').load_model()
        )

    def add_callback(self, callback: callable):
        print("[INFO] Callback agregado.")
        self.callbacks.append(callback)

    def on_cross(self, event: Dict[str, Any]):
        # Esta función ahora será el punto principal para guardar el evento inicial.
        # Es llamada directamente por CustomLineCounter para cada cruce detectado.
        event['place_code'] = self.config.get('code')
        event['stream_code'] = self.stream.get('code')
        
        # Guardar el evento inicial inmediatamente
        if self.save_person_data_to_json(event):
            print(f"[EVENT] 🚶 Persona cruzó la línea y se guardó el JSON inicial: {event.get('tid')}")

        # Añadir el evento al tracker para posible enriquecimiento, si no existe
        with self.lock:
            if event['tid'] not in self.event_tracker:
                # Copiamos el evento para no modificar directamente el que se está pasando
                event_copy = event.copy() 
                event_copy['features'] = []
                event_copy['inference_in_progress'] = False # Inicialmente no hay inferencia en progreso
                self.event_tracker[event_copy['tid']] = event_copy
                print(f"[EVENT] 🎯 Persona añadida al tracker para inferencia: {event_copy.get('tid')}")


    # on_cross_inference_gender_age ya no es estrictamente necesario como callback directo de CustomLineCounter
    # Su lógica se ha integrado en on_cross y analyze/process_face_inference_async.
    # Si aún lo necesitas por alguna razón, puedes ajustarlo, pero la estrategia es que on_cross sea el punto de entrada.
    # def on_cross_inference_gender_age(self, event: Dict[str, Any]):
    #     # La lógica de añadir al tracker y preparar para inferencia se maneja ahora en on_cross
    #     pass

    def save_person_data_to_json(self, person_data: Dict[str, Any]) -> bool:
        uuid_val = person_data.get("uuid") # Cambiado de 'uuid' a 'uuid_val' para evitar conflicto con la función uuid
        if not uuid_val:
            print("[❌] UUID faltante en los datos de la persona, no se guarda.")
            return False

        filepath = os.path.join(self.base_storage_dir, f"{uuid_val}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(person_data, f, indent=4)
            print(f"[✔] Guardado JSON exitoso: {filepath}")
            return True
        except Exception as e:
            print(f"[❌] Error al guardar JSON '{filepath}': {e}")
            return False

    def analyze(self, result: Any, frame: np.ndarray):
        face_detections = [
            d for d in result.results if d.get('label', '').lower() == 'human face'
        ]
        
        # No necesitamos tracks_to_delete aquí, se gestiona directamente en clear_event_tracker
        current_track_ids_in_result = {r['track_id'] for r in result.results if 'track_id' in r}

        # Iterar sobre los eventos actualmente en el tracker
        with self.lock:
            # Creamos una lista de TIDs activos en el tracker para evitar modificar el diccionario mientras iteramos
            tids_in_tracker_copy = list(self.event_tracker.keys())

        for track_id in tids_in_tracker_copy:
            with self.lock:
                enriched_event = self.event_tracker.get(track_id)

            if not enriched_event: # El evento pudo haber sido eliminado por otro hilo
                continue

            # Si el objeto ya no está en los resultados actuales, se gestionará en clear_event_tracker
            if track_id not in current_track_ids_in_result:
                continue

            # Solo si el evento no tiene una inferencia en progreso y no ha alcanzado el máximo de inferencias
            if not enriched_event.get('inference_in_progress', False) and \
               len(enriched_event['features']) < self.MAX_FACE_INFERENCES_PER_PERSON:
                
                # Buscar un rostro asociado a esta persona
                bbox_person = None
                for res in result.results:
                    if res.get('track_id') == track_id and res.get('label', '').lower() == 'person':
                        bbox_person = res.get('bbox')
                        break
                
                if not bbox_person: # Si no encontramos la bbox de la persona, no podemos asociar un rostro
                    continue

                center_person = self._get_center(bbox_person)
                faces_inside = [
                    f for f in face_detections if self._is_inside(f.get('bbox'), bbox_person)
                ]
                
                if not faces_inside:
                    continue

                best_face = max(faces_inside, key=lambda f: self._face_score(f['bbox'], center_person))
                x1, y1, x2, y2 = map(int, best_face['bbox'])
                face_crop = frame[y1:y2, x1:x2]

                with self.lock:
                    # Marcar que la inferencia está en progreso para este TID
                    enriched_event['inference_in_progress'] = True
                    enriched_event['inference_start_time'] = time.time() # Registrar el tiempo de inicio
                    print(f"[INFO] 🔍 Iniciando inferencia para TID {track_id}")
                self.process_face_inference_async(enriched_event, face_crop)
        
        # Llamar a clear_event_tracker con los TIDs que ya no están presentes en los resultados
        # Esto incluye aquellos que se perdieron del seguimiento o salieron del encuadre.
        tids_no_longer_tracked = set(self.event_tracker.keys()) - current_track_ids_in_result
        self.clear_event_tracker(tids_no_longer_tracked)


    def clear_event_tracker(self, stale_tids: set):
        with self.lock:
            tids_to_remove_from_tracker = set()
            for tid in stale_tids:
                event_data = self.event_tracker.get(tid)
                if event_data:
                    # Condición para guardar: ya no está siendo seguido O la inferencia se estancó.
                    # No tiene inferencia en progreso O la inferencia ha excedido el tiempo límite.
                    should_save = False
                    if not event_data.get('inference_in_progress', False):
                        should_save = True
                    elif time.time() - event_data.get('inference_start_time', 0) > self.INFERENCE_TIMEOUT_SECONDS:
                        print(f"[ALERTA] Inferencia para TID {tid} excedió el tiempo límite. Guardando estado actual.")
                        should_save = True

                    if should_save:
                        if self.save_person_data_to_json(event_data):
                            print(f"[EVENT] 🗑️ Removido y guardado TID {tid} al finalizar seguimiento o por timeout.")
                            tids_to_remove_from_tracker.add(tid)
                        else:
                            print(f"[ADVERTENCIA] No se pudo guardar JSON para TID {tid}. Se mantiene en tracker para reintentar.")
            
            # Limpiar el event_tracker eliminando los TIDs que se han procesado y guardado
            for tid in tids_to_remove_from_tracker:
                if tid in self.event_tracker:
                    del self.event_tracker[tid]


    def process_face_inference_async(self, enriched_event: Dict[str, Any], face_crop: np.ndarray):
        # Es crucial pasar una copia del evento o asegurar que el evento se maneje correctamente
        # ya que los hilos se ejecutan de forma concurrente y el objeto 'enriched_event' podría
        # ser modificado por el hilo principal o clear_event_tracker.
        # En este caso, al pasar el objeto del diccionario, si el diccionario lo mantiene, está bien.
        threading.Thread(
            target=self._process_face_inference,
            args=(enriched_event, face_crop),
            daemon=True
        ).start()

    def _process_face_inference(self, enriched_event: Dict[str, Any], face_crop: np.ndarray):
        try:
            # Realiza la inferencia del modelo
            inference = self.face_feature_model(face_crop)
            
            with self.lock:
                # Asegúrate de que el evento aún exista en el tracker antes de modificarlo
                if enriched_event['tid'] not in self.event_tracker:
                    print(f"[INFO] TID {enriched_event['tid']} ya no está en el tracker. No se actualiza.")
                    return

                # Si ya se alcanzó el máximo de inferencias, no añadimos más
                if len(enriched_event['features']) >= self.MAX_FACE_INFERENCES_PER_PERSON:
                    enriched_event['inference_in_progress'] = False # Si ya no necesitamos más, la marcamos como false
                    print(f"[INFO] TID {enriched_event['tid']} ya tenía el máximo de inferencias. No se añaden más.")
                    return

                enriched_event['features'].append(inference.results)
                enriched_event['inference_in_progress'] = False # La inferencia actual ha terminado
                print(f"[✔] Inferencia completada para TID {enriched_event['tid']}. Total features: {len(enriched_event['features'])}")

                # Si se alcanzó el máximo de inferencias después de esta, o si es la última
                if len(enriched_event['features']) >= self.MAX_FACE_INFERENCES_PER_PERSON:
                    print(f"[INFO] 🎉 TID {enriched_event['tid']} alcanzó el máximo de inferencias. Procesando y guardando.")

                    # --- Procesamiento de género ---
                    gender_scores = {'male': 0.0, 'female': 0.0}
                    for feature_list in enriched_event['features']:
                        for item in feature_list:
                            label = item.get('label', '').lower()
                            score = item.get('score', 0.0)
                            if label == 'male':
                                gender_scores['male'] += score
                            elif label == 'female':
                                gender_scores['female'] += score

                    total_gender_score = gender_scores['male'] + gender_scores['female']
                    if total_gender_score > 0:
                        confidence = max(gender_scores.values()) / total_gender_score
                        dominant_gender = max(gender_scores, key=gender_scores.get)
                        enriched_event['gender'] = dominant_gender if confidence >= 0.55 else 'neutral'
                    else:
                        enriched_event['gender'] = 'neutral'

                    # --- Procesamiento de edad ---
                    ages = []
                    for feature_list in enriched_event['features']:
                        for item in feature_list:
                            if item.get('label', '').lower() == 'age':
                                ages.append(item.get('score', 0.0))

                    ranges = [
                        (0,10), (10,15), (15, 28), (28, 40), (40, 60), (60, 100)
                    ]

                    if ages:
                        avg_age = sum(ages) / len(ages)
                        enriched_event['age'] = avg_age
                        enriched_event['age_range'] = next(
                            (r for r in ranges if r[0] <= avg_age < r[1]),
                            (0, 100)
                        )
                    else:
                        enriched_event['age'] = 0
                        enriched_event['age_range'] = (0, 100)

                    # Guardar el JSON enriquecido y eliminar del tracker
                    if self.save_person_data_to_json(enriched_event):
                        print(f"[✔] JSON enriquecido guardado para UUID {enriched_event.get('uuid')}")
                        if enriched_event['tid'] in self.event_tracker:
                            del self.event_tracker[enriched_event['tid']] # Eliminar del diccionario
                        print(f"[🧹] Evento finalizado y eliminado del tracker: {enriched_event['tid']}")

        except Exception as e:
            print(f"[❌] Error durante la inferencia de rostro para TID {enriched_event.get('tid')}: {e}")
            with self.lock:
                # Si hay un error, marcar como no en progreso para que `clear_event_tracker` pueda manejarlo
                if enriched_event['tid'] in self.event_tracker:
                    self.event_tracker[enriched_event['tid']]['inference_in_progress'] = False
                    self.event_tracker[enriched_event['tid']]['error_during_inference'] = str(e) # Registrar el error


    def _get_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_area(self, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1))

    def _face_score(self, bbox_face: Tuple[float, float, float, float], person_center: Tuple[float, float]) -> float:
        face_center = self._get_center(bbox_face)
        return self._bbox_area(bbox_face) / (1 + self._distance(face_center, person_center))

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    def _is_inside(self, small_box: Tuple[float, float, float, float], big_box: Tuple[float, float, float, float]) -> bool:
        sx1, sy1, sx2, sy2 = small_box
        bx1, by1, bx2, by2 = big_box
        return sx1 >= bx1 and sy1 >= by1 and sx2 <= bx2 and sy2 <= by2