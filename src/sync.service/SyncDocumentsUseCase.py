import os
import json
import logging
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from pymongo.errors import PyMongoError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

class SyncDocumentsUseCase:
    
    def __init__(self, directory: str):
        self.directory = directory
        
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.collection_name = 'events'
        
    def execute(self):
        logging.info("Iniciando sincronización de documentos...")

        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        if not json_files:
            logging.warning(f"No se encontraron archivos JSON en {self.directory}")
            return

        logging.info(f"Se encontraron {len(json_files)} archivos JSON en {self.directory}")

        requests = []
        for file in json_files:
            file_path = os.path.join(self.directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if "eventId" not in data:
                        logging.warning(f"Archivo {file} omitido: No tiene 'eventId'")
                        continue

                    filter_query = {"eventId": data["eventId"]}
                    requests.append(
                        UpdateOne(
                            filter_query,
                            {"$set": data},
                            upsert=True
                        )
                    )
                    logging.info(f"Preparado para insertar/actualizar: {file}")

            except Exception as e:
                logging.error(f"Error al leer {file}: {e}")

        if requests:
            try:
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                collection = db[self.collection_name]

                collection.bulk_write(requests)
                logging.info(f"Insertados/actualizados {len(requests)} documentos en la colección {self.collection_name}")
                
                client.close()
                
                for file in json_files:
                    file_path = os.path.join(self.directory, file)
                    os.remove(file_path)
                    logging.info(f"Archivo {file} eliminado.")
                
            except PyMongoError as e:
                logging.error(f"Error de conexión con MongoDB: {e}")
        else:
            logging.warning("No se encontraron documentos válidos para insertar.")

        logging.info("Finalizado el proceso de sincronización.")
