import numpy as np
import cv2
import math
from typing import List, Dict

class Tracker:
    def __init__(self, distance_threshold=50, iou_threshold=0.3, min_threshold=0.1, max_threshold=1.2, max_lost_frames=5):
        self.tracks = {}  # {id: Track}
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_lost_frames = max_lost_frames

    def update(self, detections: List[Dict], frame=None):
        
        print("Ejecutando Tracker")
        
        tracks_output = []
        if not detections:
            # Incrementar lost_frames de todos los tracks activos
            for track in self.tracks.values():
                track.lost_frames += 1
            self._remove_lost_tracks()
            return tracks_output

        # Extraer centroides y bounding boxes de las detecciones
        detection_centroids = [self._bbox_to_centroid(det['bbox']) for det in detections]
        detection_bboxes = [det['bbox'] for det in detections]
        
        # Mantener listas de detecciones y tracks no emparejados
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))

        # Asignar detecciones a tracks existentes
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_dist = float('inf')
            best_detection_idx = None
            
            for detection_idx, (det_centroid, det_bbox) in enumerate(zip(detection_centroids, detection_bboxes)):
                # Calcular la distancia y el IoU entre el track y la detección
                dist = self._calculate_distance(track.centroid, det_centroid)
                iou = self._calculate_iou(track.detection['bbox'], det_bbox)

                # Validación usando el umbral de distancia e IoU con los límites definidos
                if (self.min_threshold <= dist <= self.max_threshold or self.min_threshold <= iou <= self.max_threshold):
                    #print(f"::::::::::::::: DIST: {dist}, IOU: {iou} ::::::::::::::")
                    if iou > best_iou or (iou == best_iou and dist < best_dist):
                        #print("::::::::::::::: BEST ::::::::::::::")
                        best_iou = iou
                        best_dist = dist
                        best_detection_idx = detection_idx

            # Si se encuentra una detección adecuada, actualizar el track
            if best_detection_idx is not None:
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(best_detection_idx)
                track.update(detections[best_detection_idx])

        # Crear nuevos tracks para detecciones no emparejadas
        for detection_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[detection_idx])
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Incrementar lost_frames para tracks no emparejados
        for track_id in unmatched_tracks:
            self.tracks[track_id].lost_frames += 1

        # Eliminar tracks perdidos
        self._remove_lost_tracks()
        # Generar salida de tracks
        for track in self.tracks.values():
            tracks_output.append(track)
            if frame is not None:
                # Dibujar información del track en el frame
                x1, y1, x2, y2 = map(int, track.detection['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (253, 6, 105), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 245, 144), 2)

        return tracks_output

    def _remove_lost_tracks(self):
        self.tracks = {track_id: track for track_id, track in self.tracks.items()
                       if track.lost_frames <= self.max_lost_frames}

    def _bbox_to_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy

    def _calculate_distance(self, track_centroid, det_centroid):
        return math.hypot(track_centroid[0] - det_centroid[0], track_centroid[1] - det_centroid[1])

    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)

        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    

class Track:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.detection = detection
        self.centroid = self._bbox_to_centroid(self.detection['bbox'])
        self.lost_frames = 0

    def update(self, detection):
        self.detection['bbox'] = detection['bbox']
        self.centroid = self._bbox_to_centroid(self.detection['bbox'])
        self.lost_frames = 0

    def _bbox_to_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy
