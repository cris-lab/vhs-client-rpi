import time
import logging
import asyncio
from src.AnalyzeDetectionsUseCase import AnalyzeDetectionsUseCase

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

async def main():
    logging.info("Iniciando servicio de análisis de detecciones faciales...")
    sync_use_case = AnalyzeDetectionsUseCase(
        "/var/lib/vhs/detections",
        "/var/lib/vhs/events"
    )

    while True:
        # Ejecutar la función asincrónica
        await sync_use_case.execute()
        logging.info("Esperando 15 minutos para la próxima ejecución...")
        await asyncio.sleep(60)  # 900 segundos = 15 minutos

if __name__ == "__main__":
    # Ejecutar la función main asincrónica
    asyncio.run(main())
