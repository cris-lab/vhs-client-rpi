import os
import logging
import numpy as np
import json
import shutil
from PIL import Image

from src.AgeEstimationService import AgeEstimationService
from src.GenderClassificationService import GenderClassificationService

class AnalyzeDetectionsUseCase:

    def __init__(self, directory: str, directory_out: str):
        self.directory = directory
        self.directory_out = directory_out
        self.ageEstimationService = AgeEstimationService()
        self.genderClassificationService = GenderClassificationService()

    async def execute(self):
        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        if not json_files:
            logging.warning(f"No se encontraron archivos JSON en {self.directory}")
            return

        logging.info(f"Se encontraron {len(json_files)} archivos JSON en {self.directory}")

        for file in json_files:
            file_path = os.path.join(self.directory, file)
            file_id = file.split(".")[0]
            image_path = os.path.join(self.directory, file_id + ".jpg")
            updated_json_path = os.path.join(self.directory_out, file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get('eventType') != "in":
                    logging.info(f"Evento {data.get('eventType')} ignorado para archivo {file}")
                    continue

                if not os.path.exists(image_path):
                    logging.warning(f"No se encontró la imagen {image_path}")
                    continue

                try:
                    pil_image = Image.open(image_path)
                    frame = np.array(pil_image)
                except Exception as e:
                    logging.error(f"Error al abrir/convertir imagen {image_path}: {e}")
                    continue

                try:
                    response_age = await self.ageEstimationService.execute(frame)
                    response_gender = await self.genderClassificationService.execute(frame)
                    data['analisis'] = {
                        'age': response_age['results'][0],
                        'gender': response_gender['results']
                    }
                except Exception as e:
                    logging.error(f"Error en análisis para {image_path}: {e}")
                    continue

                # Guardar JSON actualizado
                with open(updated_json_path, 'w', encoding='utf-8') as out_file:
                    json.dump(data, out_file, ensure_ascii=False, indent=4)

                logging.info(f"Análisis guardado y actualizado: {updated_json_path}")

                # Borrar imagen y JSON original solo si todo salió bien
                os.remove(image_path)
                logging.info(f"Imagen eliminada: {image_path}")

                os.remove(file_path)
                logging.info(f"JSON original eliminado: {file_path}")

            except Exception as e:
                logging.error(f"Error general al procesar {file}: {e}")
