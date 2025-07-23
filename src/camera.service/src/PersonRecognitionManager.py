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

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.person_data = {}
        self.lost_tracks_buffer = {}
        self.cleanup_track_timeout_sec_interval = 2
        self.lost_track_cleanup_timeout_sec = 10
        self.base_storage_dir = config.get('base_storage_dir', '/opt/vhs/storage/detections')
        self.face_save_limit = config.get('roi_save_limit', 3)
        self.roi_padding_factor = config.get('roi_padding_factor', 0.15)
        self.fece_feature_model = degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1').load_model(),
        )
        self.embedding_model = ModelLoader('repvgg_a0_person_reid--256x128_quant_hailort_hailo8l_1').load_model()
        os.makedirs(self.base_storage_dir, exist_ok=True)

    def process_tracks(self, frame, result):
        now = time.time()
        current_frame_track_ids = set()
        face_detections = [d for d in result.results if d.get('label', '').lower() == 'human face']

        for track in result.results:
            if track.get('label') != 'head':
                continue
            
            track_id = track.get('track_id', None)
            if track_id is None:
                continue
            current_frame_track_ids.add(track_id)
            
            if track_id not in self.person_data:
                new_uuid = str(uuid.uuid4())
                new_person_data = {
                    "uuid": new_uuid,
                    "gender": None,
                    "age": None,
                    "features": [],
                    "description": "",
                    "attributes": {},
                    "frames_seen": 1,
                    "duration_tracked": 0.0,
                    "total_movement": 0,
                    "first_position": track.get('bbox', []),
                    "last_position": track.get('bbox', []),
                    "first_appearance_time": now,
                    "last_seen": now,
                    "lost_since": None,
                    "origin_id": track_id,
                    "reid_id": track_id,
                    "captured_rois": [],
                    "best_roi_path": None,
                    "valid_track": True,
                    "event_log": ["detected"],
                    "trails": result.trails.get(track_id, [])
                }
                
                reassigned_id = self.attempt_visual_reid(frame, track.get('bbox', []))
                
                if reassigned_id is not None:
                    recovered_data = self.lost_tracks_buffer.pop(reassigned_id)
                    
                    new_person_data['uuid'] = recovered_data.get('uuid', new_uuid)
                    new_person_data['reid_id'] = recovered_data.get('reid_id', new_person_data['reid_id'])
                    new_person_data['origin_id'] = recovered_data.get('origin_id', track_id)
                    new_person_data['captured_rois'] = recovered_data.get('captured_rois', [])
                    new_person_data['event_log'] += ["recovered"]
                    
                    print(f"[ReID] Track {track_id} reassigned and recovered from {reassigned_id}. New reid_id: {new_person_data['reid_id']}.")
                
                self.person_data[track_id] = new_person_data

                roi = self.extract_roi(frame, track.get('bbox', []))
                if roi is not None:
                    embedding_data = self.embedding_model(roi)
                    if self.debug:
                        print('Generated embedding data type:', type(embedding_data))
                    
                    if isinstance(embedding_data, dict) and 'data' in embedding_data:
                        self.person_data[track_id]['last_roi_image'] = embedding_data['data']
                    elif isinstance(embedding_data, list):
                        self.person_data[track_id]['last_roi_image'] = embedding_data
                    else:
                        if self.debug:
                            print(f"Error: Unexpected embedding format for track {track_id}")
                        self.person_data[track_id]['last_roi_image'] = []

            else:
                info = self.person_data[track_id]
                info['last_seen'] = now
                info['last_position'] = track.get('bbox', [])
                info['trails'] = result.trails.get(track_id, [])
                info['frames_seen'] += 1
                info['duration_tracked'] = info['last_seen'] - info['first_appearance_time']
                info['lost_since'] = None

            self.process_faces(frame, track, track_id, face_detecciones)

        self.handle_lost_and_cleanup_tracks(current_frame_track_ids, now)
        return self.person_data

    def handle_lost_and_cleanup_tracks(self, current_frame_track_ids, now):
        for track_id in list(self.person_data.keys()):
            if track_id not in current_frame_track_ids:
                if track_id not in self.lost_tracks_buffer:
                    self.move_track_to_lost(track_id, now)

        self.clean_up_lost_tracks(now)

    def compare_embeddings(self, emb1, emb2):
        if emb1 is None or emb2 is None or not emb1 or not emb2:
            return float('inf')
        
        emb1_array = np.array(emb1).flatten()
        emb2_array = np.array(emb2).flatten()
        
        if emb1_array.size == 0 or emb2_array.size == 0:
            return float('inf')
        
        distance = cosine(emb1_array, emb2_array)
        return distance

    def attempt_visual_reid(self, frame, bbox):
        roi_current = self.extract_roi(frame, bbox)
        if roi_current is None:
            return None
        
        embedding_data = self.embedding_model(roi_current)
        current_embedding = []

        if isinstance(embedding_data, dict) and 'data' in embedding_data:
            current_embedding = embedding_data['data']
        elif isinstance(embedding_data, list):
            current_embedding = embedding_data
        
        if not current_embedding:
            return None
        
        best_match_id = None
        min_distance = self.config.get('reid_distance_threshold', 0.5)

        for lost_id, lost_data in list(self.lost_tracks_buffer.items()):
            saved_embedding = lost_data.get('last_roi_image', None)
            
            distance = self.compare_embeddings(current_embedding, saved_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match_id = lost_id
    
        return best_match_id

    def head_has_face(self, head_bbox, face_detections, iou_threshold=0.3):
        if not face_detections:
            return False
        for face in face_detections:
            face_bbox = face.get('bbox', [0, 0, 0, 0])
            if self.calculate_iou(head_bbox, face_bbox) >= iou_threshold:
                return True
        return False

    def calculate_iou(self, box1, box2):
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

    def process_faces(self, frame, head_track, track_id, face_detecciones):
        if head_track.get('label') != 'head':
            return
        head_bbox = head_track.get('bbox', [0, 0, 0, 0])
        if not self.head_has_face(head_bbox, face_detecciones):
            if self.debug:
                print(f"Skipping ROI for track {track_id} (UUID: {self.person_data[track_id]['uuid'] if track_id in self.person_data else 'N/A'}) - No face detected within head bbox.")
            return 
        roi = self.extract_roi(frame, head_bbox)
        if roi is not None:
            person_info = self.person_data.get(track_id)
            if not person_info:
                return
            if len(person_info['features']) < self.face_save_limit:
                face_result = self.fece_feature_model(roi)
                person_info['features'].append(face_result.results)

    def get_roi_area(self, roi):
        return roi.shape[0] * roi.shape[1]

    def get_roi_area_from_file(self, img_path):
        if not img_path or not os.path.exists(img_path):
            return 0
        img = cv2.imread(img_path)
        if img is not None:
            return img.shape[0] * img.shape[1]
        return 0

    def move_track_to_lost(self, track_id, now):
        if track_id in self.person_data:
            self.lost_tracks_buffer[track_id] = self.person_data.pop(track_id)
            self.lost_tracks_buffer[track_id]['last_seen'] = now
            self.lost_tracks_buffer[track_id]['lost_since'] = now
            self.lost_tracks_buffer[track_id]['event_log'].append("lost")
            print(f"Moved track {track_id} (UUID: {self.lost_tracks_buffer[track_id]['uuid']}) to lost_tracks_buffer (awaiting final cleanup check).")

    def is_false_positive(self, track_data):
        trails = track_data.get('trails', [])
        track_id = track_data.get('origin_id')
        if not isinstance(trails, list) or len(trails) < 2:
            if self.debug:
                print(f"Track {track_id}: Very short or empty trajectory (len={len(trails)}).")
            return True
        movimiento_total = self.calculate_trail_movement(trails)
        if movimiento_total < 20:
            if self.debug:
                print(f"Track {track_id}: Total movement ({movimiento_total}) less than 20.")
            return True
        duracion = track_data['last_seen'] - track_data['first_appearance_time']
        if duracion < 0.5:
            if self.debug:
                print(f"Track {track_id}: Duration ({duracion:.2f}s) less than 0.5s.")
            return True
        if self.debug:
            print(f"Track {track_id}: Passes false positive verification (Movement: {movimiento_total}, Duration: {duracion:.2f}s).")
        return False

    def calculate_trail_movement(self, trails):
        if not trails or len(trails) < 2:
            return 0
        first_point = trails[0]
        last_point = trails[-1]
        if not isinstance(first_point, (list, tuple)) or len(first_point) < 4:
            if self.debug: print("Invalid format for first_point in trail.")
            return 0
        if not isinstance(last_point, (list, tuple)) or len(last_point) < 4:
            if self.debug: print("Invalid format for last_point in trail.")
            return 0
        dx = abs(last_point[0] - first_point[0])
        dy = abs(last_point[1] - first_point[1])
        movimiento_total = dx + dy
        return movimiento_total

    def bbox_center(self,bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return (0, 0)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def clean_up_lost_tracks(self, now):
        lost_timeout = self.lost_track_cleanup_timeout_sec
        tracks_to_delete = []

        for track_id in list(self.lost_tracks_buffer.keys()):
            track_data = self.lost_tracks_buffer[track_id]
            lost_since = track_data['lost_since']
            time_difference = now - lost_since
            person_uuid = track_data.get("uuid", "unknown")

            trails = track_data.get('trails', [])
            track_data['duration_tracked'] = track_data['last_seen'] - track_data['first_appearance_time']

            if trails and len(trails) > 1:
                track_data['total_movement'] = self.calculate_trail_movement(trails)
                start_x, start_y = self.bbox_center(trails[0])
                end_x, end_y = self.bbox_center(trails[-1])
                track_data['positions_summary'] = {
                    "start_bbox": trails[0],
                    "end_bbox": trails[-1],
                    "start": [start_x, start_y],
                    "end": [end_x, end_y],
                    "count": len(trails)
                }
                start_zone = self.map_to_grid_zone(start_x, start_y)
                end_zone = self.map_to_grid_zone(end_x, end_y)
                track_data['entry_zone'] = start_zone
                track_data['exit_zone'] = end_zone
                track_data['direction'] = self.estimate_direction(start_x, start_y, end_x, end_y)
            else:
                track_data['total_movement'] = 0
                track_data['positions_summary'] = None
                track_data['entry_zone'] = None
                track_data['exit_zone'] = None
                track_data['direction'] = None

            ages, genders = [], []
            for feature_set in track_data.get('features', []):
                for item in feature_set:
                    if item.get("label") == "Age":
                        ages.append(item.get("score"))
                    elif item.get("label") == "Male":
                        genders.append(item.get("score"))
            track_data["age"] = float(np.mean(ages)) if ages else None
            track_data["gender"] = float(np.mean(genders)) if genders else None

            if time_difference > lost_timeout:
                if not self.is_false_positive(track_data):
                    track_data['valid_track'] = True
                    track_data['event_log'].append("finalized")
                else:
                    track_data['valid_track'] = False
                    track_data['event_log'].append("discarded_fp")
                    
                self.save_person_data_to_json_async(track_data)
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            del self.lost_tracks_buffer[track_id]

    def map_to_grid_zone(self, x, y, grid_size=8, frame_size=640):
        cell_size = frame_size / grid_size
        col = int(x / cell_size)    
        row = int(y / cell_size)
        return f"Z{row}{col}"

    def estimate_direction(self, start_x, start_y, end_x, end_y):
        dx = end_x - start_x
        dy = end_y - start_y
        if abs(dx) > abs(dy):
            return "East" if dx > 0 else "West"
        else:
            return "South" if dy > 0 else "North"
            
    def save_person_data_to_json_async(self, person_data):
        threading.Thread(
            target=self.save_person_data_to_json,
            args=(person_data,),
            daemon=True
        ).start()
    
    def save_person_data_to_json(self, person_data):
        person_uuid = person_data.get("uuid")
        if not person_uuid:
            print(f"Error: No UUID found for track data. Cannot save JSON.")
            return

        json_filename = os.path.join(self.base_storage_dir, f"{person_uuid}.json")
        data_to_save = person_data.copy()
        
        try:
            if 'last_roi_image' in data_to_save:
                del data_to_save['last_roi_image']
            
            with open(json_filename, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Person data saved to: {json_filename}")
        except Exception as e:
            print(f"Error saving person data to JSON for UUID {person_uuid}: {e}")