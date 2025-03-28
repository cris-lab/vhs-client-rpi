import uuid
import time
import json
import cv2  # Usado para guardar el frame como imagen

class Event:
    
    def __init__(self, 
        store_code=None,
        event_type='detection', 
        description='',
        object_type='person'
    ):
        self.storeCode      = store_code
        self.eventId        = uuid.uuid4()
        self.description    = description
        self.eventType      = event_type
        self.objectType     = object_type
        self.timeStamp      = int(time.time() * 1000)
      
    def save(self, frame):
        # Guardar el frame como imagen con el nombre del eventId.jpg
        image_filename = f"storage/detections/{self.eventId}.jpg"
        cv2.imwrite(image_filename, frame)  # Usamos OpenCV para guardar la imagen
        print(f"Frame guardado como {image_filename}")
        
        # Crear un diccionario con los datos de la clase para el archivo JSON
        event_data = {
            "storeCode": self.storeCode,
            "description": self.description,
            "eventId": str(self.eventId),
            "eventType": self.eventType,
            "objectType": self.objectType,
            "timeStamp": self.timeStamp
        }
        
        # Guardar los datos en un archivo JSON con el nombre del eventId.json
        json_filename = f"storage/detections/{self.eventId}.json"
        with open(json_filename, 'w') as json_file:
            json.dump(event_data, json_file, indent=4)
        
        print(f"Datos guardados como {json_filename}")
