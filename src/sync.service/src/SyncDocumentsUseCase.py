import os
import json
import logging
import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv('/opt/vhs/src/setup.service/.env')

class SyncDocumentsUseCase:

    def __init__(self, directory: str):
        self.directory = directory
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.collection_name = 'detections'
 
    def execute(self):
        logging.info("Iniciando sincronización de documentos...")

        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        if not json_files:
            logging.warning(f"No se encontraron archivos JSON en {self.directory}")
            return

        logging.info(f"Se encontraron {len(json_files)} archivos JSON en {self.directory}")

        requests_bulk = []
        data_by_file = {}

        for file in json_files:
            file_path = os.path.join(self.directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    data_by_file[file] = data 

                    filter_query = {"uuid": data["uuid"]}
                    requests_bulk.append(
                        UpdateOne(
                            filter_query,
                            {"$set": data},
                            upsert=True
                        )
                    )
                    logging.info(f"Preparado para insertar/actualizar: {file}")

            except Exception as e:
                logging.error(f"Error al leer {file}: {e}")

        if requests_bulk:
            try:
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                collection = db[self.collection_name]

                collection.bulk_write(requests_bulk)
                logging.info(f"Insertados/actualizados {len(requests_bulk)} documentos en la colección {self.collection_name}")

                client.close()

            except PyMongoError as e:
                logging.error(f"Error de conexión con MongoDB: {e}")
        else:
            logging.warning("No se encontraron documentos válidos para insertar.")

        logging.info("Finalizado el proceso de sincronización.")
