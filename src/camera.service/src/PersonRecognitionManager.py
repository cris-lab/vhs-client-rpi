import time
import uuid
import json
import os
import sys
import degirum_tools
import threading
import numpy as np
import cv2
from src.ModelLoader import ModelLoader
from scipy.spatial.distance import cosine

class PersonRecognitionManager:

    def __init__(self, config, stream, debug=False):
        self.config = config
        self.stream = stream
        self.debug = debug
        self.person_data = {}
        self.lost_tracks_buffer = {}
        self.cleanup_track_timeout_sec_interval = 2
        self.lost_track_cleanup_timeout_sec = 10
        self.base_storage_dir = config.get('base_storage_dir', '/opt/vhs/storage/detections')
        self.face_save_limit = config.get('roi_save_limit', 3)
        self.roi_padding_factor = config.get('roi_padding_factor', 0.15)
        self.face_feature_model = degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1').load_model(),
        )
        os.makedirs(self.base_storage_dir, exist_ok=True)

        self.frame_counter = 0 
        
        # --- PARÁMETROS BÁSICOS Y CONTADORES GLOBALES ---
        self.frame_width = config.get('frame_width', 640)  
        self.frame_height = config.get('frame_height', 480) 
        self.frame_rate = config.get('frame_rate', 30)    
        
        self.exit_border_threshold_px = config.get('exit_border_threshold_px', 50) 
        self.min_consistent_frames_for_exit = config.get('min_consistent_frames_for_exit', 5) 
        self.min_movement_per_frame_for_exit = config.get('min_movement_per_frame_for_exit', 5) 

        # Contadores globales de eventos (para el dashboard)
        self.global_entry_count = 0
        self.global_inferred_exits = { "North": 0, "South": 0, "East": 0, "West": 0, "Total": 0 }
        self.last_inferred_exit_info_global = None 

        # --- DICCIONARIOS PARA ETIQUETAS MÁS AMIGABLES (Internos para la lógica) ---
        self._direction_labels = {
            "North": "Norte", "South": "Sur", "East": "Este", "West": "Oeste",
            "NorthEast": "Noreste", "SouthEast": "Sureste", "NorthWest": "Noroeste", "SouthWest": "Suroeste",
            "Static": "Estático"
        }
        # --------------------------------------------------------------------------
        
    def process_tracks(self, frame, result):
        """
        Procesa las trazas de objetos detectadas en un fotograma.
        Actualiza el estado de las personas y sus historiales.
        """
        now = time.time()
        self.frame_counter += 1 
        current_frame_track_ids = set()
        face_detections = [d for d in result.results if d.get('label', '').lower() == 'human face']

        for track in result.results:
            if track.get('label') not in self.stream.get('tracker', {}).get('class_list', []):
                continue
            
            track_id = track.get('track_id', None)
            if track_id is None:
                continue
            current_frame_track_ids.add(track_id)
            
            head_bbox = track.get('bbox', [])
            
            if track_id not in self.person_data:
                new_uuid = str(uuid.uuid4())
                new_person_data = {
                    "uuid": new_uuid,
                    "gender": None,
                    "age": None,
                    "features": [], # Se mantienen para el cálculo de género/edad, no en el JSON final si no se pide.
                    "frames_seen": 1,
                    "duration_tracked": 0.0,
                    "total_movement": 0,
                    "first_position": head_bbox,
                    "last_position": head_bbox,
                    "first_appearance_time": now,
                    "last_seen": now,
                    "lost_since": None, # Se mantiene para el cálculo del tiempo perdido.
                    "origin_id": track_id,
                    "valid_track": True,
                    "event_log": ["detected"],
                    "trails": result.trails.get(track_id, []), # Se mantiene para cálculos, no en JSON final.
                    "zone_history": [] # Se mantiene para cálculos, no en JSON final.
                }
                
                self.person_data[track_id] = new_person_data
                
                if new_person_data['valid_track']: 
                    self.global_entry_count += 1

            else:
                info = self.person_data[track_id]
                info['last_seen'] = now
                info['last_position'] = head_bbox
                info['trails'] = result.trails.get(track_id, [])
                info['frames_seen'] += 1
                info['duration_tracked'] = info['last_seen'] - info['first_appearance_time']
                info['lost_since'] = None
            
            self.process_faces(frame, track, track_id, face_detections)

        self.handle_lost_and_cleanup_tracks(current_frame_track_ids, now)
        return self.person_data

    def handle_lost_and_cleanup_tracks(self, current_frame_track_ids, now):
        """Gestiona las trazas que ya no se detectan y activa la limpieza."""
        for track_id in list(self.person_data.keys()):
            if track_id not in current_frame_track_ids:
                if track_id not in self.lost_tracks_buffer:
                    self.move_track_to_lost(track_id, now)

        self.clean_up_lost_tracks(now)

    def head_has_face(self, head_bbox, face_detections, iou_threshold=0.3):
        """Verifica si un bounding box de cabeza contiene una detección de rostro basada en IoU."""
        if not face_detections:
            return False
        for face in face_detections:
            face_bbox = face.get('bbox', [0, 0, 0, 0])
            if self.calculate_iou(head_bbox, face_bbox) >= iou_threshold:
                return True
        return False

    def calculate_iou(self, box1, box2):
        """Calcula la Intersección sobre Unión (IoU) de dos bounding boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - intersection_area)
        return intersection_area / union_area

    def extract_roi(self, frame, bbox):
        """Extrae una Región de Interés (ROI) del fotograma con padding."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * self.roi_padding_factor)
        pad_y = int(height * self.roi_padding_factor)
        x1_padded = int(x1 - pad_x)
        y1_padded = int(y1 - pad_y)
        x2_padded = int(x2 + pad_x)
        y2_padded = int(y2 + pad_y)
        h, w, _ = frame.shape
        x1_final = max(0, x1_padded)
        y1_final = max(0, y1_padded)
        x2_final = min(w, x2_padded)
        y2_final = min(h, y2_padded)
        if x2_final <= x1_final or y2_final <= y1_final:
            return None
        roi = frame[y1_final:y2_final, x1_final:x2_final]
        if roi.size == 0:
            return None
        return roi

    def process_faces(self, frame, head_track, track_id, face_detections):
        """Procesa las detecciones de rostro dentro de las trazas de cabeza para extraer características."""
        if head_track.get('label') != 'head':
            return
        
        head_bbox = head_track.get('bbox', [0, 0, 0, 0])
        if not self.head_has_face(head_bbox, face_detections):
            if self.debug:
                print(f"Skipping ROI for track {track_id} (UUID: {self.person_data[track_id]['uuid'] if track_id in self.person_data else 'N/A'}) - No face detected within head bbox.")
            return 
        roi = self.extract_roi(frame, head_bbox)
        if roi is not None:
            person_info = self.person_data.get(track_id)
            if not person_info:
                return
            # if len(person_info['features']) < self.face_save_limit:
            #     face_result = self.face_feature_model(roi)
            #     person_info['features'].append(face_result.results)
    
    def get_roi_area(self, roi):
        """Calcula el área de una ROI."""
        return roi.shape[0] * roi.shape[1]

    def get_roi_area_from_file(self, img_path):
        """Lee una imagen desde un archivo y calcula su área."""
        if not img_path or not os.path.exists(img_path):
            return 0
        img = cv2.imread(img_path)
        if img is not None:
            return img.shape[0] * img.shape[1]
        return 0

    def move_track_to_lost(self, track_id, now):
        """Mueve una traza activa al buffer de trazas perdidas."""
        if track_id in self.person_data:
            track_data = self.person_data.pop(track_id)
            
            if self.is_false_positive(track_data):
                if self.debug:
                    print(f"Track {track_id} es un falso positivo, descartando.")
                return 

            self.lost_tracks_buffer[track_id] = track_data
            self.lost_tracks_buffer[track_id]['last_seen'] = now
            self.lost_tracks_buffer[track_id]['lost_since'] = now
            track_data['event_log'].append("lost") 
            if self.debug:
                print(f"Moved track {track_id} (UUID: {self.lost_tracks_buffer[track_id]['uuid']}) to lost_tracks_buffer.")


    def is_false_positive(self, track_data):
        """Determina si una traza es probablemente un falso positivo basándose en el movimiento, duración y fotogramas vistos."""
        trails = track_data.get('trails', [])
        track_id = track_data.get('origin_id')
        if not isinstance(trails, list) or len(trails) <= 2:
            if self.debug:
                print(f"Track {track_id}: Trayectoria muy corta o vacía (len={len(trails)}).")
            return True
        movimiento_total = self.calculate_trail_movement(trails)
        if movimiento_total < 20:
            if self.debug:
                print(f"Track {track_id}: Movimiento total ({movimiento_total}) menor a 20.")
            return True
        duracion = track_data['last_seen'] - track_data['first_appearance_time']
        if duracion < 0.5:
            if self.debug:
                print(f"Track {track_id}: Duración ({duracion:.2f}s) menor a 0.5s.")
            return True
        frames_seen = track_data.get('frames_seen', 1)
        if frames_seen < 3:
            if self.debug:
                print(f"Track {track_id}: Fotogramas vistos ({frames_seen}) menor a 3.")
            return True 
        if self.debug:
            print(f"Track {track_id}: Pasa la verificación de falso positivo (Movimiento: {movimiento_total}, Duración: {duracion:.2f}s).")
        return False

    def calculate_trail_movement(self, trails):
        """Calcula el movimiento total en píxeles de una traza."""
        if not trails or len(trails) < 2:
            return 0
        first_point = trails[0]
        last_point = trails[-1]
        if not isinstance(first_point, (list, tuple)) or len(first_point) < 4:
            if self.debug: print("Formato inválido para first_point en trail.")
            return 0
        if not isinstance(last_point, (list, tuple)) or len(last_point) < 4:
            if self.debug: print("Formato inválido para last_point en trail.")
            return 0
        dx = abs(last_point[0] - first_point[0])
        dy = abs(last_point[1] - first_point[1])
        movimiento_total = dx + dy
        return movimiento_total

    def bbox_center(self,bbox):
        """Calcula las coordenadas centrales de un bounding box."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return (0, 0)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # --- FUNCIONES AUXILIARES PARA LA INFERENCIA DE SALIDA ---
    def get_recent_direction_and_speed(self, trails, num_frames):
        """
        Calcula la dirección dominante y la velocidad promedio en los últimos N fotogramas de una traza.
        Retorna: (dominant_direction, avg_speed_px_per_sec) o (None, None)
        """
        if len(trails) < num_frames:
            return None, None # No hay suficientes datos para un análisis consistente

        recent_trails = trails[-num_frames:]
        
        displacements = []
        for i in range(1, len(recent_trails)):
            p1_center_x, p1_center_y = self.bbox_center(recent_trails[i-1])
            p2_center_x, p2_center_y = self.bbox_center(recent_trails[i])
            dx = p2_center_x - p1_center_x
            dy = p2_center_y - p1_center_y
            displacements.append((dx, dy))

        if not displacements:
            return None, None

        total_dx = sum(d[0] for d in displacements)
        total_dy = sum(d[1] for d in displacements)
        
        dominant_direction = None
        # Mejorar la estimación de la dirección para incluir diagonales
        if abs(total_dx) < 1 and abs(total_dy) < 1: # Si el movimiento es insignificante
            dominant_direction = "Static"
        elif abs(total_dx) > abs(total_dy) * 2: # Principalmente horizontal
            dominant_direction = "East" if total_dx > 0 else "West"
        elif abs(total_dy) > abs(total_dx) * 2: # Principalmente vertical
            dominant_direction = "South" if total_dy > 0 else "North"
        else: # Movimiento diagonal
            if total_dy < 0: # Norte
                dominant_direction = "NorthEast" if total_dx > 0 else "NorthWest"
            else: # Sur
                dominant_direction = "SouthEast" if total_dx > 0 else "SouthWest"
        
        total_distance = sum(np.sqrt(d[0]**2 + d[1]**2) for d in displacements)
        
        duration_sec = (num_frames - 1) / self.frame_rate if self.frame_rate > 0 else 0
        avg_speed_px_per_sec = total_distance / duration_sec if duration_sec > 0 else 0

        return dominant_direction, avg_speed_px_per_sec

    def is_close_to_frame_border(self, bbox, threshold_px):
        """
        Verifica si un bbox está cerca de los bordes del frame.
        Retorna True si está cerca y una lista de las direcciones de los bordes cercanos (ej. "North", "East").
        """
        x1, y1, x2, y2 = bbox
        
        is_close = False
        border_directions = []

        if x1 <= threshold_px:
            is_close = True
            border_directions.append("West")
        if x2 >= self.frame_width - threshold_px:
            is_close = True
            border_directions.append("East")
        if y1 <= threshold_px:
            is_close = True
            border_directions.append("North")
        if y2 >= self.frame_height - threshold_px:
            is_close = True
            border_directions.append("South")
            
        return is_close, border_directions

    def estimate_direction(self, start_x, start_y, end_x, end_y):
        """Estima la dirección cardinal o diagonal entre dos puntos."""
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Considerar movimiento insignificante como "Static"
        if abs(dx) < 5 and abs(dy) < 5: 
            return "Static"

        angle_rad = np.arctan2(-dy, dx) # -dy porque Y aumenta hacia abajo
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        # Definir rangos para las 8 direcciones
        if 22.5 <= angle_deg < 67.5:
            return "NorthEast"
        elif 67.5 <= angle_deg < 112.5:
            return "North"
        elif 112.5 <= angle_deg < 157.5:
            return "NorthWest"
        elif 157.5 <= angle_deg < 202.5:
            return "West"
        elif 202.5 <= angle_deg < 247.5:
            return "SouthWest"
        elif 247.5 <= angle_deg < 292.5:
            return "South"
        elif 292.5 <= angle_deg < 337.5:
            return "SouthEast"
        else: # 337.5 a 360 y 0 a 22.5
            return "East"

    # -----------------------------------------------------------------------

    def clean_up_lost_tracks(self, now):
        """
        Limpia las trazas que se han perdido por demasiado tiempo,
        guardando sus datos enriquecidos.
        """
        lost_timeout = self.lost_track_cleanup_timeout_sec
        tracks_to_delete = []

        for track_id in list(self.lost_tracks_buffer.keys()):
            track_data = self.lost_tracks_buffer[track_id]
            lost_since = track_data['lost_since']
            time_difference = now - lost_since
            person_uuid = track_data.get("uuid", "unknown")

            trails = track_data.get('trails', [])
            track_data['duration_tracked'] = track_data['last_seen'] - track_data['first_appearance_time']

            # --- CÁLCULO DE POSICIÓN, RESUMEN ---
            first_pos_center = None
            last_pos_center = None
            
            if trails and len(trails) > 1:
                track_data['total_movement'] = self.calculate_trail_movement(trails)
                
                first_pos_bbox = trails[0]
                first_pos_center = self.bbox_center(first_pos_bbox)
                
                last_pos_bbox = trails[-1]
                last_pos_center = self.bbox_center(last_pos_bbox)

                track_data['positions_summary'] = {
                    "start_bbox": trails[0],
                    "end_bbox": trails[-1],
                    "start": [first_pos_center[0], first_pos_center[1]],
                    "end": [last_pos_center[0], last_pos_center[1]],
                    "count": len(trails)
                }

                track_data['direction'] = self.estimate_direction(first_pos_center[0], first_pos_center[1], last_pos_center[0], last_pos_center[1])
                track_data['direction_label'] = self._direction_labels.get(track_data['direction'], track_data['direction'])
                
            else: # Para trazas muy cortas o sin movimiento significativo
                track_data['total_movement'] = 0
                track_data['positions_summary'] = None
                track_data['direction'] = "Static" 
                track_data['direction_label'] = self._direction_labels.get("Static", "Static")

            # --- CÁLCULO DE GÉNERO Y EDAD ---
            ages, genders = [], []
            for feature_set in track_data.get('features', []):
                for item in feature_set:
                    if item.get("label") == "Age":
                        ages.append(item.get("score"))
                    elif item.get("label") == "Male":
                        genders.append(item.get("score"))
            
            track_data["age"] = float(np.mean(ages)) if ages else None
            track_data["gender"] = float(np.mean(genders)) if genders else None
            if track_data["gender"] is not None:
                track_data["gender_label"] = "Male" if track_data["gender"] > 0.5 else "Female"
            else:
                track_data["gender_label"] = "Unknown"


            # --- LÓGICA DE INFERENCIA DE SALIDA POR DIRECCIÓN Y PROXIMIDAD A BORDE ---
            inferred_exit_type = None 
            
            if not self.is_false_positive(track_data) and len(trails) >= self.min_consistent_frames_for_exit:
                
                recent_dominant_direction, avg_speed = self.get_recent_direction_and_speed(
                    trails, 
                    self.min_consistent_frames_for_exit
                )
                
                last_bbox = track_data['last_position'] 
                is_close, close_borders = self.is_close_to_frame_border(
                    last_bbox, 
                    self.exit_border_threshold_px
                )

                if is_close and \
                   recent_dominant_direction is not None and \
                   avg_speed is not None and \
                   avg_speed > (self.min_movement_per_frame_for_exit / (1/self.frame_rate)): 
                    
                    direction_matches_border = False
                    for border_key in close_borders: 
                        if border_key == recent_dominant_direction: 
                            direction_matches_border = True
                            inferred_exit_type = border_key # Guarda la dirección cardinal
                            break
                        elif len(close_borders) == 2: 
                            # Si está en una esquina y la dirección es diagonal hacia afuera de esa esquina
                            if (("North" in close_borders and "East" in close_borders and recent_dominant_direction == "NorthEast") or
                                ("North" in close_borders and "West" in close_borders and recent_dominant_direction == "NorthWest") or
                                ("South" in close_borders and "East" in close_borders and recent_dominant_direction == "SouthEast") or
                                ("South" in close_borders and "West" in close_borders and recent_dominant_direction == "SouthWest")):
                                direction_matches_border = True
                                inferred_exit_type = recent_dominant_direction # Guarda la dirección diagonal
                                break
                    
                    if direction_matches_border:
                        track_data['inferred_exit_type'] = inferred_exit_type
                        if self.debug:
                            print(f"Track {track_id} (UUID: {person_uuid}) INFERIDA SALIDA por {inferred_exit_type} (última pos. cerca de borde, mov. {recent_dominant_direction}, velocidad {avg_speed:.2f} px/s).")
            # --------------------------------------------------------------------------

            # --- PREPARACIÓN DEL CAMPO `exit_classification` PARA EL JSON FINAL ---
            # Este campo es para tu reportería global de Entradas/Salidas.
            exit_classification = "Finalized_Normal_Loss" # Default si no es salida ni FP
            
            if track_data['valid_track']:
                # Si se infirió una salida por movimiento hacia un borde:
                if track_data.get('inferred_exit_type'):
                    # La etiqueta en el JSON será "Inferred_Exit_North", "Inferred_Exit_East", etc.
                    # Usamos la dirección amigable para la etiqueta final
                    friendly_direction = self._direction_labels.get(track_data['inferred_exit_type'], track_data['inferred_exit_type'])
                    exit_classification = f"Inferred_Exit_{friendly_direction}"

            track_data['exit_classification'] = exit_classification
            # -----------------------------------------------------------------------


            if time_difference > lost_timeout:
                event_to_log = "finalized_normal_loss" # Estado por defecto para event_log
                
                if not self.is_false_positive(track_data):
                    track_data['valid_track'] = True
                    
                    if track_data.get('inferred_exit_type'):
                        # event_log usará la dirección cardinal inferida
                        event_to_log = f"inferred_exit_{track_data['inferred_exit_type']}" 
                        
                        # Actualizar contadores globales de salidas inferidas
                        direction_for_count = track_data['inferred_exit_type'] # Es una cardinal o diagonal
                        
                        # Queremos contar solo las cardinales para el dashboard simplificado
                        if direction_for_count in ["North", "South", "East", "West"]:
                            self.global_inferred_exits[direction_for_count] += 1
                            self.global_inferred_exits["Total"] += 1
                        elif direction_for_count in ["NorthEast", "SouthEast", "NorthWest", "SouthWest", "Static"]:
                            # Las diagonales y estáticos se cuentan solo en el total si no hay un contador específico
                            self.global_inferred_exits["Total"] += 1
                            if self.debug:
                                print(f"Advertencia: Dirección de salida inferida '{direction_for_count}' no mapeada en contadores globales individuales.")
                        else: # Si es una dirección completamente inesperada
                            self.global_inferred_exits["Total"] += 1
                            if self.debug:
                                print(f"Advertencia: Dirección de salida inferida inesperada '{direction_for_count}'.")
                        
                        self.last_inferred_exit_info_global = {
                            "uuid": person_uuid,
                            "direction": self._direction_labels.get(track_data['inferred_exit_type'], track_data['inferred_exit_type']), 
                            "time": time.strftime("%H:%M:%S", time.localtime(now))
                        }
                else:
                    track_data['valid_track'] = False
                    event_to_log = "discarded_fp"
                    
                track_data['event_log'].append(event_to_log)
                
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            final_track_data = self.lost_tracks_buffer.pop(track_id)
            if self.debug:
                print(f"Finalized and removed track {track_id} (UUID: {final_track_data['uuid']}) from lost_tracks_buffer.")

    def get_global_counts(self):
        """Retorna los contadores globales de entradas y salidas inferidas."""
        return {
            "global_entry_count": self.global_entry_count,
            "global_inferred_exits": self.global_inferred_exits,
            "last_inferred_exit_info_global": self.last_inferred_exit_info_global
        }