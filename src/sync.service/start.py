import time
import logging
from src.SyncDocumentsUseCase import SyncDocumentsUseCase

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    
    logging.info("Iniciando servicio de sincronización de documentos")
    sync_use_case = SyncDocumentsUseCase("/opt/vhs/storage/detections")
    
    while True:
        sync_use_case.execute()
        logging.info("Ciclo de sincronización completado, esperando 1 segundo antes de la próxima ejecución")
        time.sleep(1000) 