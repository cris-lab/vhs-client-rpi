import os
import logging
import numpy as np
import json
import shutil
from PIL import Image
from io import BytesIO

from src.AgeEstimationService import AgeEstimationService
from src.GenderClassificationService import GenderClassificationService

class AnalyzeDetectionsUseCase:
    
    def __init__(self, directory: str, directory_out: str):
        self.directory = directory
        self.directory_out = directory_out
        self.ageEstimationService = AgeEstimationService()
        self.genderClassificationService = GenderClassificationService()

    async def execute(self):
        # Obtener archivos JSON
        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        if not json_files:
            logging.warning(f"No se encontraron archivos JSON en {self.directory}")
            return

        logging.info(f"Se encontraron {len(json_files)} archivos JSON en {self.directory}")

        for file in json_files:
            
            file_path = os.path.join(self.directory, file)
            try:
                # Leer el archivo JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    
                    file_id = file.split(".")[0]
                    data    = json.load(f)
                    # Ruta de la imagen
                    image_path = os.path.join(self.directory, file_id + ".jpg")
                    
                    if(data['eventType'] == "in"):
                    
                        if not os.path.exists(image_path):
                            logging.warning(f"No se encontró la imagen {image_path}")
                            continue

                        # Abrir la imagen y convertirla en numpy.ndarray
                        try:
                            pil_image = Image.open(image_path)
                            frame = np.array(pil_image) # Convertir la imagen PIL a numpy.ndarray
                        except Exception as e:
                            logging.error(f"Error al abrir o convertir la imagen {image_path}: {e}")
                            continue
                    
                        # Realizar análisis de edad y género
                        try:
                            
                            response_age = await self.ageEstimationService.execute(frame)
                            response_gender = await self.genderClassificationService.execute(frame)
                            
                            analisis = {
                                'age': response_age['results'][0],
                                'gender': response_gender['results']
                            }
                            
                        except Exception as e:
                            logging.error(f"Error en la estimación de edad o clasificación de género para {image_path}: {e}")
                            continue

                        # Agregar análisis al JSON
                        data['analisis'] = analisis

                        # Guardar el archivo JSON actualizado
                        updated_json_path = os.path.join(self.directory_out, file)
                        with open(updated_json_path, 'w', encoding='utf-8') as out_file:
                            json.dump(data, out_file, ensure_ascii=False, indent=4)
                        logging.info(f"Análisis guardado para {file} en {self.directory_out}")


                    # Eliminar la imagen
                    os.remove(image_path)
                    logging.info(f"Imagen eliminada: {image_path}")

                    # Mover el archivo JSON a directory_out
                    shutil.move(updated_json_path, os.path.join(self.directory_out, file))
                    logging.info(f"Archivo JSON movido a {self.directory_out}")
                    
                    os.remove(file_path)
                    logging.info(f"Archivo JSON original eliminado: {file_path}")
                    
                    

            except Exception as e:
                logging.error(f"Error al procesar el archivo {file}: {e}")

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)