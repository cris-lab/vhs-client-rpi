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

                    filter_query = {"eventId": data["eventId"]}
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

                # Enviar a la API y eliminar archivos
                for file, data in data_by_file.items():
                    camera_id = data.get("camera", {}).get("id")
                    if not camera_id:
                        logging.warning(f"No se encontró camera.id en {file}")
                        continue

                    try:
                        response = requests.post(
                            f"{self.api_url_base}/{camera_id}",
                            json=data,
                            timeout=5
                        )
                        if response.status_code == 200:
                            logging.info(f"Evento enviado correctamente a la API para cámara {camera_id}")
                            os.remove(os.path.join(self.directory, file))
                            logging.info(f"Archivo {file} eliminado.")
                        else:
                            logging.error(f"Error al enviar {file} a la API. Status: {response.status_code}")
                    except Exception as e:
                        logging.error(f"Error al enviar {file} a la API: {e}")

                client.close()

            except PyMongoError as e:
                logging.error(f"Error de conexión con MongoDB: {e}")
        else:
            logging.warning("No se encontraron documentos válidos para insertar.")

        logging.info("Finalizado el proceso de sincronización.")
