import requests, os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class AgeEstimationService:
    
    url = "http://127.0.0.1:5002/inference"
    
    async def execute(self, frame): 
        """
        Detecta objetos en un solo frame usando un servicio REST.

        Args:
            frame (numpy.ndarray): Imagen del frame a procesar.

        Returns:
            Dict: Respuesta JSON del servicio REST con las detecciones.
        """
        # Convertir el numpy.ndarray a una imagen PIL
        pil_image = Image.fromarray(frame.astype('uint8'))  # Convertir a uint8 si es necesario
        byte_io = BytesIO()
        pil_image.save(byte_io, format='JPEG')  # Guardar la imagen en formato JPEG
        byte_io.seek(0)
        
        # Preparar los archivos para la solicitud
        files = {'file': ('frame.jpg', byte_io, 'image/jpeg')}
        
        print("POST::Image to AgeEstimationService")
        params = {}

        # Hacer la solicitud POST al servicio REST
        response = requests.post(self.url, files=files, params=params)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code != 200:
            raise Exception(f"Error en la solicitud REST: {response.status_code}, {response.text}")
        
        # Obtener la respuesta en formato JSON
        return response.json()