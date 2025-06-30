import time
import cv2
import numpy as np
from typing import List
from src.Event import Event
from src.Label import Label
from src.Detection import Detection

class InOutUseCase():
    
    tracked_objects_state = {} # Para almacenar el estado de cada objeto
    
    def __init__(self, stream=None, tracker=None, cross_line=None, centroid_orientation=None, counter_interpolation=0):
        
        self.stream = stream
        self.tracker = tracker
        self.cross_line = cross_line  # Línea de cruce, debe ser una lista de 2 puntos (x1, y1), (x2, y2)
        self.centroid_orientation = centroid_orientation
        self.counter = {"IN": 0, "OUT": 0}
        self.counter_interpolation = counter_interpolation
    

    def execute(self, frame, place, detections: List[Detection] = None):
        
        print("Ejecutando In Out Use Case")

        # label = Label(
        #     text=f"IN: {self.counter['IN']} | OUT: {self.counter['OUT']}",
        #     font=cv2.FONT_HERSHEY_SIMPLEX,
        #     font_scale=0.7,
        #     color=(255, 255, 255),
        #     thickness=2,
        #     padding=10,
        #     background_color=(253, 6, 105),
        #     position=(30, 30)
        # )
        
        # label.draw(frame)
        # Inicializar el resumen fuera del bucle de áreas
        timestamp = time.time()
        # Dibujar la línea de cruce
        if self.cross_line is not None:
            cv2.line(frame, tuple(self.cross_line[0]), tuple(self.cross_line[1]), (30, 245, 144), 2)

        # Usamos el tracker para actualizar las posiciones de los objetos detectados
        tracked_objects = self.tracker.update(detections, frame)
        print(f"Tracked objects: {len(tracked_objects)}")
        #print(f"Tracked objects: {tracked_objects}")
        #print(f"Tracked objects: {len(tracked_objects)}")
        # Recorrer los objetos rastreados y etiquetarlos en el frame
        for tracked_obj in tracked_objects:
            # print(f"Detection type: {type(tracked_obj.detection)}")
            # print(f"Detection content: {tracked_obj.detection}")
            track_id = int(tracked_obj.track_id)
            bbox = tracked_obj.detection['bbox'] 
            #print(f"Track ID: {track_id}, BBox: {bbox}")
            x1, y1, x2, y2 = map(int, bbox)
            center = self.get_centroid(x1, y1, x2, y2)
            
            #cv2.circle(frame, center, 3, (80, 30, 245), -1)

            # Verificar si el objeto ha cruzado la línea de cruce
            if self.cross_line:
                crossing = self.check_crossing(track_id, center)
                if crossing['has_crossed']:
                    Event(
                        camera={
                            "id":   self.stream['id'],
                            "name": self.stream['name'],
                            "code": self.stream['code']
                        },
                        place=place,
                        object_type=tracked_obj.detection['label'].lower(), 
                        event_type=crossing["direction"].lower()
                    ).save(frame)
                    
        #cv2.line(frame, tuple(self.cross_line[0]), tuple(self.cross_line[1]), (30, 245, 144), 2)
        
        return frame, tracked_objects
                    
        
    def get_centroid(self, x1, y1, x2, y2):
        
        config = self.centroid_orientation
        config_x = config['horizontal']
        config_y = config['vertical']
        
        if(config_x == 'center'):
            cx = (x1 + x2) // 2
        if(config_x == 'left'):
            cx = x1
        if(config_x == 'right'):
            cx = x2
        if(config_y == 'top'):
            cy = y1
        if(config_y == 'middle'):
            cy = (y1 + y2) // 2
        if(config_y == 'bottom'):
            cy = y2
            
        return cx, cy
        

    def check_crossing(self, track_id, center):
        """
        Verifica si el objeto ha cruzado la línea de cruce en dirección vertical (de abajo a arriba o de arriba a abajo).
        Elimina el seguimiento del objeto después de cruzar la línea.
        """
        if track_id not in self.tracked_objects_state:
            self.tracked_objects_state[track_id] = {"last_position": center, "has_crossed": False, "direction": None}

        obj_state = self.tracked_objects_state[track_id]
        
        # Obtener la última posición
        last_position = obj_state["last_position"]
        
        # Obtener los puntos de la línea (suponemos que es horizontal)
        line_start = self.cross_line[0]
        line_end = self.cross_line[1]
        
        # Verificar si la línea es vertical
        if line_start[0] == line_end[0]:
            # Línea vertical, solo compararemos en el eje Y (arriba a abajo o abajo a arriba)
            if last_position[1] < line_start[1] and center[1] >= line_start[1]:
                # El objeto ha cruzado hacia abajo (de arriba a abajo)
                if not obj_state["has_crossed"]:
                    obj_state["has_crossed"] = True
                    obj_state["direction"] = "OUT" if self.counter_interpolation == 1 else "IN"
                    self.remove_tracking(track_id)  # Eliminar el objeto después de cruzar
            elif last_position[1] >= line_start[1] and center[1] < line_start[1]:
                # El objeto ha cruzado hacia arriba (de abajo a arriba)
                if not obj_state["has_crossed"]:
                    obj_state["has_crossed"] = True
                    obj_state["direction"] = "IN" if self.counter_interpolation == 1 else "OUT"
                    self.remove_tracking(track_id)  # Eliminar el objeto después de cruzar
        else:
            # Aquí solo consideramos el cruce en el eje Y
            if last_position[1] < line_start[1] and center[1] >= line_start[1]:
                # El objeto ha cruzado hacia abajo (de arriba a abajo)
                if not obj_state["has_crossed"]:
                    obj_state["has_crossed"] = True
                    obj_state["direction"] = "OUT" if self.counter_interpolation == 1 else "IN"
                    self.remove_tracking(track_id)  # Eliminar el objeto después de cruzar
            elif last_position[1] >= line_start[1] and center[1] < line_start[1]:
                # El objeto ha cruzado hacia arriba (de abajo a arriba)
                if not obj_state["has_crossed"]:
                    obj_state["has_crossed"] = True
                    obj_state["direction"] = "IN" if self.counter_interpolation == 1 else "OUT"
                    self.remove_tracking(track_id)  # Eliminar el objeto después de cruzar

        # Verificar si ya se registró un cruce
        if obj_state["has_crossed"] and obj_state["direction"]:
            self.counter[obj_state["direction"]] += 1

        # Actualizar la última posición del objeto
        obj_state["last_position"] = center

        return obj_state


    def remove_tracking(self, track_id):
        """
        Elimina el objeto del seguimiento y borra su estado.
        """
        if track_id in self.tracked_objects_state:
            del self.tracked_objects_state[track_id]
        print(f"Objeto {track_id} eliminado del seguimiento.")
