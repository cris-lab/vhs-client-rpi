from datetime import datetime
import degirum_tools
from src.ModelLoader import ModelLoader
from src.PersonRecognitionManager import PersonRecognitionManager
import numpy as np
import cv2
import src.utils as vhs_utils

class FrameProcessor:
    
    def __init__(self, config, stream, class_list=[]):

        self.stream             = stream
        self.tracker            = degirum_tools.ObjectTracker(
            class_list=stream.get('tracker', {}).get('class_list', class_list),
            track_thresh=stream.get('tracker', {}).get('track_thresh', 0.5),
            track_buffer= stream.get('tracker', {}).get('track_buffer', 30),
            match_thresh= stream.get('tracker', {}).get('trail_depth', 20),
            trail_depth=stream.get('tracker', {}).get('trail_depth', 20),
            anchor_point=degirum_tools.AnchorPoint.TOP_CENTER,
            annotation_color= (255, 0, 0),
        )

        self.combined_model = degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8n_relu6_human_head--640x640_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_face--640x640_quant_hailort_hailo8l_1').load_model(),
        )
        

        self.person_recognition_manager = PersonRecognitionManager(config)


    def execute(self, frame):
        result = self.combined_model(frame)

        if not result:
            return frame, False

        self.filtrar_detecciones_validas(result.results)

        if not result:
            return frame, False
        
        if len(result.results) > 0:
            self.tracker.analyze(result)
        
        person_data = self.person_recognition_manager.process_tracks(frame, result)
        
        # Lógica para pintar de otro color cuando se hace match
        for person in person_data.values():
            if 'last_position' in person and len(person['last_position']) == 4:
                # Si el reid_id es diferente del origin_id, significa que se ha recuperado un track.
                # En ese caso, pintamos de un color diferente, por ejemplo, rojo (0,0,255).
                # Si no, usamos un color por defecto, como el verde (0,255,0).
                if person['origin_id'] != person['reid_id']:
                    color = (0, 0, 255)  # Color rojo para match (re-identificado)
                else:
                    color = (0, 255, 0)  # Color verde para track nuevo o normal

                self.draw_bbox_with_id(frame, person['last_position'], person['reid_id'], color)
                
        if self.stream.get('grid').get('show', False):
            
            grid_size = self.stream.get('grid').get('size', 4)
            vhs_utils.draw_grid_on_frame(frame, grid_size, color=(229, 225, 232), thickness=1)
        
        return frame, True
    
    
    
    def filtrar_detecciones_validas(self, result):
        """
        Filtra las detecciones inválidas directamente sobre la lista original.
        Elimina cualquier objeto que no tenga 'bbox' y 'label'.

        Args:
            result (list): Lista de detecciones (modificada in-place).
        """
        indices_a_eliminar = []

        for idx, detection in enumerate(result):
            if not isinstance(detection, dict):
                #print(f"[WARNING] Resultado no es un diccionario válido: {detection}")
                indices_a_eliminar.append(idx)
                continue

            if 'bbox' not in detection or 'label' not in detection:
                #print(f"[WARNING] Detección descartada por estar incompleta: {detection}")
                indices_a_eliminar.append(idx)

        for idx in reversed(indices_a_eliminar):
            del result[idx]


    def draw_bbox_with_id(self, frame, bbox, track_id, color=(0, 255, 0)):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = str(track_id)

        font_scale = 0.5
        font_thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        box_width = label_width + 10
        box_height = label_height + 8

        box_x1 = x2 - box_width
        box_y1 = y1
        box_x2 = x2
        box_y2 = y1 + box_height

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, -1)

        text_x = box_x1 + 5
        text_y = box_y2 - 5
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        return frame