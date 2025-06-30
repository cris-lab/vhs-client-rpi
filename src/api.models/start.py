import argparse
import json
import uvicorn
import io
import os
import logging
import numpy as np
import degirum._zoo_accessor as zoo
import sys
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.concurrency import run_in_threadpool # <<<< Importante: Para ejecutar código bloqueante

import degirum as dg

# Configuración de logging para capturar los errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Función para obtener los parámetros de la línea de comandos
def parse_arguments():
    parser = argparse.ArgumentParser(description="Servicio de inferencia con FastAPI")
    parser.add_argument('--model',  type=str, required=True, help='Nombre del modelo a cargar')
    parser.add_argument('--port',   type=int, required=True, help='Puerto en el que correrá el servidor FastAPI')
    parser.add_argument('--width',  type=int, required=True, help='Ancho en el que la imagen será redimensionada para ingresar al modelo')
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
model_path = Path(f"/opt/vhs/src/api.models/models/{model_name}/{model_name}.json")

# Cargar el modelo (esta parte es síncrona, se ejecuta una vez al inicio)
try:
    
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {model_path}")

    accessor = zoo._LocalInferenceSingleFileZooAccessor(str(model_path))
    # Cargar el modelo (nombre sin extensión)
    model = accessor.load_model(model_path.stem)
    # Configurar el dispositivo local (HAILO)
    model.device_type = ['HAILORT/HAILO8L']
    model.inference_host_address = "@local"
    model.measure_time = True
    
    print("✅ Modelo cargado exitosamente.")
    print(f"Nombre del modelo: {model._model_name}")
    
    logger.info(f"Modelo '{model_name}' cargado exitosamente en Hailo.")

except Exception as e:
    logger.critical(f"ERROR: No se pudo cargar el modelo '{model_name}': {e}", exc_info=True)
    sys.exit(1) # Salir si el modelo no se puede cargar, ya que la API no funcionará.

app = FastAPI()

# Función auxiliar para el pre-procesamiento de la imagen (síncrona, para ThreadPool)
def _preprocess_image_for_model(image_data: bytes, target_width: int, target_height: int):
    """
    Realiza el pre-procesamiento de la imagen (Letterbox) para el modelo.
    Devuelve la imagen NumPy lista para el modelo y los parámetros de post-procesamiento.
    """
    image = Image.open(io.BytesIO(image_data))

    ancho_original, alto_original = image.size

    escala = min(target_width / ancho_original, target_height / alto_original)

    nuevo_ancho = int(ancho_original * escala)
    nuevo_alto = int(alto_original * escala)

    image_resized = image.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    lienzo = Image.new("RGB", (target_width, target_height), (128, 128, 128))
    pad_x = (target_width - nuevo_ancho) // 2
    pad_y = (target_height - nuevo_alto) // 2
    lienzo.paste(image_resized, (pad_x, pad_y))

    image_listo_para_modelo = np.array(lienzo)

    return image_listo_para_modelo, pad_x, pad_y, escala, ancho_original, alto_original


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    try:
        # --- PASO 1: Leer la imagen (operación asíncrona de E/S, no bloquea) ---
        image_data = await file.read()

        # --- PASO 2: Pre-procesamiento (se ejecuta en un hilo para no bloquear el event loop) ---
        image_listo_para_modelo, pad_x, pad_y, escala, ancho_original, alto_original = \
            await run_in_threadpool(_preprocess_image_for_model, image_data, args.width, args.height)

        # --- PASO 3: Inferencia (se ejecuta en un hilo para no bloquear el event loop) ---
        result = await run_in_threadpool(model, image_listo_para_modelo)

        # Imprimir estadísticas de tiempo si es necesario
        logger.info(f"Hailo Inference Stats: {model._time_stats.__str__()}")

        inference_results = getattr(result, '_inference_results', [])

        if not inference_results or inference_results == [{}]:
            return JSONResponse(content={"results": [], "message": "No se detectaron objetos."}, status_code=200)

        # --- PASO 4: Post-procesamiento para compatibilidad hacia atrás ---
        # Este post-procesamiento es ligero en comparación con la inferencia/pre-procesamiento,
        # así que puede ejecutarse en el event loop principal sin problema.
        for res in inference_results:
            xmin_crudo, ymin_crudo, xmax_crudo, ymax_crudo = res['bbox']

            xmin_sin_pad = xmin_crudo - pad_x
            ymin_sin_pad = ymin_crudo - pad_y
            xmax_sin_pad = xmax_crudo - pad_x
            ymax_sin_pad = ymax_crudo - pad_y

            xmin_final_pixel = xmin_sin_pad / escala
            ymin_final_pixel = ymin_sin_pad / escala
            xmax_final_pixel = xmax_sin_pad / escala
            ymax_final_pixel = ymax_sin_pad / escala

            # Asegúrate de que las coordenadas no sean negativas y no excedan las dimensiones originales
            xmin_final_pixel = max(0.0, xmin_final_pixel)
            ymin_final_pixel = max(0.0, ymin_final_pixel)
            xmax_final_pixel = min(float(ancho_original), xmax_final_pixel)
            ymax_final_pixel = min(float(alto_original), ymax_final_pixel)


            bbox_corregido = [xmin_final_pixel, ymin_final_pixel, xmax_final_pixel, ymax_final_pixel]
            res['bbox'] = bbox_corregido

        # --- PASO 5: Devolver la respuesta ---
        result_json = jsonable_encoder(inference_results)

        return JSONResponse(content={"results": result_json}, status_code=200)

    except Exception as e:
        logger.error(f"Error en la inferencia: {str(e)}", exc_info=True)
        os._exit(1)

if __name__ == "__main__":
    try:
        logger.info(f"Iniciando la aplicación de inferencia.")
        logger.info(f"Modelo: {model_name}")
        logger.info(f"Puerto: {args.port}")
        logger.info(f"Dimensiones de redimensionamiento: {args.width}x{args.height}")

        uvicorn.run(app, host="0.0.0.0", port=args.port)

    except Exception as e:
        logger.critical(f"Error fatal al iniciar la aplicación: {e}", exc_info=True)
        sys.exit(1) # <<<< Importante: Indicar un fallo en el inicio