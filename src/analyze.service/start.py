import time
import logging
import asyncio
import os
from src.AnalyzeDetectionsUseCase import AnalyzeDetectionsUseCase

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

async def main():
    logging.info("Iniciando servicio de análisis de detecciones faciales...")
    
    detections_folder = "/var/lib/vhs/detections"
    events_folder = "/var/lib/vhs/events"
    
    sync_use_case = AnalyzeDetectionsUseCase(detections_folder, events_folder)

    while True:
        # Consultar la carpeta de detecciones
        detections_files = os.listdir(detections_folder)

        if detections_files:
            logging.info(f"Archivos encontrados para procesar: {detections_files}")
            
            # Ejecutar el análisis para los archivos encontrados
            await sync_use_case.execute()
            
            logging.info("Eventos procesados y enviados.")
        else:
            logging.info("No se encontraron archivos para procesar.")
        
        # Esperar 1 segundo antes de volver a consultar la carpeta
        logging.info("Esperando 1 segundo para la próxima consulta...")
        await asyncio.sleep(1)  # 1 segundo entre cada consulta

if __name__ == "__main__":
    # Ejecutar la función main asincrónica
    asyncio.run(main())
