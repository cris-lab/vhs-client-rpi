from datetime import datetime
import degirum_tools
from src.ModelLoader import ModelLoader
from src.PersonRecognitionManager import PersonRecognitionManager
import numpy as np
import cv2

class FrameProcessor:
    
    def __init__(self, config, stream, class_list=['head']):

        self.stream             = stream
        self.tracker            = degirum_tools.ObjectTracker(
            class_list=class_list,
            track_thresh=0.3,
            track_buffer=20,
            match_thresh=0.85,
            trail_depth=10,
            anchor_point=degirum_tools.AnchorPoint.TOP_CENTER,
            annotation_color= (255, 0, 0),
        )

        #self.combined_model = ModelLoader('yolov8n_relu6_human_head--640x640_quant_hailort_hailo8l_1').load_model()
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
        
        # Mejorar esto en otra version
        for person in person_data.values():
            if 'bbox' in person and len(person['bbox']) == 4:
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {person['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            if 'age' in person:
                cv2.putText(frame, f"Age: {person['age']}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            
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

        # Elimina desde el final para no romper los índices
        for idx in reversed(indices_a_eliminar):
            del result[idx]

        # if not result:
        #     print("[WARNING] Todas las detecciones fueron descartadas.")

