import os
import json
import logging
import requests # No se usa en el código proporcionado, pero se mantiene si es parte de un contexto más amplio.
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
        data_by_file = {} # Esto ahora almacenará el nombre del archivo si es necesario para la eliminación.
        files_to_delete_on_success = [] # Nueva lista para almacenar las rutas de los archivos a eliminar

        for file in json_files:
            file_path = os.path.join(self.directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # No necesitamos almacenar la data, solo la ruta del archivo si el documento es válido
                    files_to_delete_on_success.append(file_path)

                    filter_query = {"uuid": data["uuid"]}
                    requests_bulk.append(
                        UpdateOne(
                            filter_query,
                            {"$set": data},
                            upsert=True
                        )
                    )
                    logging.info(f"Preparado para insertar/actualizar: {file}")

            except json.JSONDecodeError as e: # Mejorar el manejo de errores para JSON inválido
                logging.error(f"Error: El archivo {file} no es un JSON válido. Detalles: {e}")
                # Si el archivo no es un JSON válido, no lo añadimos a files_to_delete_on_success
                if file_path in files_to_delete_on_success:
                    files_to_delete_on_success.remove(file_path) # Asegurarse de que no se elimine si hubo error
            except Exception as e:
                logging.error(f"Error inesperado al leer {file}: {e}")
                if file_path in files_to_delete_on_success:
                    files_to_delete_on_success.remove(file_path) # Asegurarse de que no se elimine si hubo error


        if requests_bulk:
            try:
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                collection = db[self.collection_name]

                result = collection.bulk_write(requests_bulk)
                logging.info(f"Insertados/actualizados {result.upserted_count} documentos nuevos y modificados {result.modified_count} en la colección {self.collection_name}")
                
                # --- INICIO DEL CÓDIGO AÑADIDO PARA ELIMINAR ARCHIVOS ---
                if files_to_delete_on_success:
                    logging.info("Iniciando eliminación de archivos JSON procesados exitosamente...")
                    for file_path in files_to_delete_on_success:
                        try:
                            os.remove(file_path)
                            logging.info(f"Archivo eliminado: {file_path}")
                        except OSError as e:
                            logging.error(f"Error al eliminar el archivo {file_path}: {e}")
                else:
                    logging.warning("No hay archivos para eliminar después del procesamiento exitoso.")
                # --- FIN DEL CÓDIGO AÑADIDO PARA ELIMINAR ARCHIVOS ---

                client.close()

            except PyMongoError as e:
                logging.error(f"Error de conexión con MongoDB o error en bulk_write: {e}")
            except Exception as e:
                logging.error(f"Error inesperado durante la operación de base de datos o eliminación de archivos: {e}")
        else:
            logging.warning("No se encontraron documentos válidos para insertar.")

        logging.info("Finalizado el proceso de sincronización.")