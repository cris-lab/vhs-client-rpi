import requests, os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime, time # <<< Añadir
from zoneinfo import ZoneInfo     # <<< Añadir (o from pytz import timezone si usas pytz)

load_dotenv()

class DetectionService:
    
    url = "http://127.0.0.1:5000/inference"
    
    def __init__(self, url=None, time_zone=None, schedule=None):
        """
        Inicializa el servicio de detección de objetos.

        Args:
            url (str): URL del servicio REST para la detección de objetos.
        """
        if url:  
            self.url = url
            
        # Establecer la zona horaria.  Si no se proporciona, usar UTC como predeterminado.
        self.time_zone = ZoneInfo(time_zone) if time_zone else ZoneInfo('UTC')

        self.schedule = schedule if schedule else {
            'enabled': True,
            'days': ['*'],
            'start_time': '00:00',
            'end_time': '23:59',
        }
        
    def _is_within_schedule(self):
        """
        Verifica si la hora actual está dentro del horario de detección configurado.

        Returns:
            bool: True si está dentro del horario, False en caso contrario.
        """
        if not self.schedule['enabled']:
            return True  # Si el horario está desactivado, siempre está dentro del horario.

        now = datetime.now(self.time_zone)
        current_time = now.time()
        current_day = now.strftime("%A").lower()  # Obtener el nombre del día en minúsculas (ej. "monday")

        # Verificar si el día actual está en la lista de días permitidos o si se permiten todos los días ('*').
        if '*' not in self.schedule['days'] and current_day not in self.schedule['days']:
            return False
 
        start_time = time.fromisoformat(self.schedule['start_time'])
        end_time = time.fromisoformat(self.schedule['end_time'])

        # Manejar el caso donde el horario cruza la medianoche (ej. 22:00 - 02:00)
        if start_time <= end_time:
            return start_time <= current_time <= end_time
        else:
            # El horario cruza la medianoche
            return start_time <= current_time or current_time <= end_time

    
    async def execute(self, frame): 
        """
        Detecta objetos en un solo frame usando un servicio REST.

        Args:
            frame (numpy.ndarray): Imagen del frame a procesar.

        Returns:
            Dict: Respuesta JSON del servicio REST con las detecciones.
        """
        
        if not self._is_within_schedule():
            print("Detección fuera del horario programado.")
            return {
                'result': [],
                'status': True,
                'message': 'Detección omitida: fuera del horario programado.'
            }
        
        # Convertir el numpy.ndarray a una imagen PIL
        pil_image = Image.fromarray(frame.astype('uint8'))  # Convertir a uint8 si es necesario
        byte_io = BytesIO()
        pil_image.save(byte_io, format='JPEG')  # Guardar la imagen en formato JPEG
        byte_io.seek(0)
        
        # Preparar los archivos para la solicitud
        files = {'file': ('frame.jpg', byte_io, 'image/jpeg')}
        
        #print("POST::Detecting objects in frame...")
        params = {}

        try:
            # Hacer la solicitud POST al servicio REST
            response = requests.post(self.url, files=files, params=params)
            # Verificar si la solicitud fue exitosa
            if response.status_code != 200:
                raise Exception(f"Error en la solicitud REST: {response.status_code}, {response.text}")
            # Obtener la respuesta en formato JSON
            json = response.json()
            json['status'] = True
            
            return json
    
        except Exception as e:
            
            return {
                'result': [],
                'status': False,
                'message': f'Error en el servicio {self.url}: {str(e)}'
            }