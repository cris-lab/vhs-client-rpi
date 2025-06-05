import argparse
import degirum as dg
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json, uvicorn, io, os
from PIL import Image
import numpy as np
import logging

# Configuración de logging para capturar los errores
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función para obtener los parámetros de la línea de comandos
def parse_arguments():
    parser = argparse.ArgumentParser(description="Servicio de inferencia con FastAPI")
    parser.add_argument('--model', type=str, required=True, help='Nombre del modelo a cargar')
    parser.add_argument('--port', type=int, required=True, help='Puerto en el que correrá el servidor FastAPI')
    parser.add_argument('--width', type=int, required=True, help='Ancho en el que la imagen será redimensionada para ingresar al modelo')
    parser.add_argument('--height', type=int, required=True, help='Alto en el que la imagen será redimensionada para ingresar al modelo')
    return parser.parse_args()

# Parsear los argumentos
args = parse_arguments()

# Parámetros de configuración
inference_host_address = "@local"
device_type = ['HAILORT/HAILO8L']
zoo_url = "degirum/hailo"

# Cargar el modelo con el nombre proporcionado en los argumentos
model_name = args.model

# Cargar el modelo
model = dg.load_model(
    model_name=model_name,
    device_type=device_type,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url
)

model.measure_time = True

app = FastAPI()

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    try:
        # Leer la imagen recibida
        image_data = await file.read()
        # Convertir la imagen a un objeto PIL
        image = Image.open(io.BytesIO(image_data))
        # Redimensionar la imagen a 640x640
        image = image.resize((args.width, args.height))
        # Convertir la imagen a RGB si no lo es
        image = image.convert("RGB")
        # Convertir la imagen a un array de numpy
        image = np.array(image)
        # Realizar la inferencia con el modelo
        result = model(image)  # Esto depende de cómo espera los datos el modelo
        
        print(model._time_stats.__str__())
        # Convertir los resultados a formato JSON
        # Acceder a los resultados
        inference_results = getattr(result, '_inference_results', None)

        # Si no hay resultados válidos
        if not inference_results or inference_results == [{}]:
            return JSONResponse(content={"results": [], "message": "No se detectaron objetos."}, status_code=200)

        # Codificar resultados en JSON
        result_json = jsonable_encoder(inference_results)

        # Retornar resultados
        return JSONResponse(content={"results": result_json}, status_code=200)

    except Exception as e:
        # Capturar y registrar el error
        logger.error(f"Error en la inferencia: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
        exit(0)

if __name__ == "__main__":
    try:
        
        logger.info(f"Iniciando la aplicación en el puerto {args.port}")
        logger.info(f"Modelo cargado: {model_name}")
        logger.info(f"Ancho de redimensión de imagen: {args.width}")
        logger.info(f"Alto de redimensión de imagen: {args.height}")
        
        # Iniciar el servidor FastAPI usando uvicorn, con el puerto proporcionado en los argumentos
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        
    except Exception as e:
        # Capturar y registrar el error si ocurre
        logger.error(f"Error al iniciar la aplicación: {e}")
        exit(0)