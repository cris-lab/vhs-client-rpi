import time
import logging
from SyncDocumentsUseCase import SyncDocumentsUseCase

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    
    logging.info("Iniciando servicio de sincronización de documentos")
    sync_use_case = SyncDocumentsUseCase("/var/lib/vhs/events")
    
    while True:
        sync_use_case.execute()
        logging.info("Esperando 15 minutos para la próxima ejecución...")
        time.sleep(60)  # 900 segundos = 15 minutos