from pathlib import Path # Necesario para manejar rutas de archivos
import sys
import logging # Asumiendo que usas logging, si no, puedes usar print o configurar uno básico

# CAMBIO: Usar degirum._zoo_accessor como se proporcionó
import degirum._zoo_accessor as zoo
from degirum.exceptions import DegirumException

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelLoader:
    # CAMBIO: El constructor ahora recibe solo el nombre del modelo
    def __init__(self, model_name: str):
        """
        Inicializa el cargador de un modelo HailoRT específico.

        Args:
            model_name (str): El nombre del modelo a cargar (ej. 'yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8l_1').
        """
        self.model_name = model_name

        if not self.model_name:
            logger.critical("El nombre del modelo no puede estar vacío.")
            sys.exit(1)

        # Configuración del dispositivo Hailo (puede ser fija o leerse de la config global si aplica)
        self.device_type = ['HAILORT/HAILO8L'] 
        self.inference_host_address = "@local" 
        
        # Construir la ruta absoluta al modelo
        self.model_path = Path(f"/opt/vhs/models/{self.model_name}/{self.model_name}.json")

        self.loaded_model = None # Aquí se almacenará el modelo cargado

    def load_model(self):
        """
        Carga el modelo HailoRT especificado en el constructor.

        Returns:
            object: El objeto del modelo cargado de HailoRT.
        """
        try:
            # --- Cargar el Modelo ---
            if not self.model_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo del modelo: {self.model_path}")
            
            # Usar self.model_path y self.model_name
            accessor = zoo._LocalInferenceSingleFileZooAccessor(str(self.model_path))
            self.loaded_model = accessor.load_model(self.model_name)
            self.loaded_model.device_type = self.device_type
            self.loaded_model.inference_host_address = self.inference_host_address
            self.loaded_model.measure_time = True # Asumiendo que esta propiedad existe y es relevante
            logger.info(f"Modelo '{self.model_name}' cargado con éxito.")

        except FileNotFoundError as fnfe:
            logger.critical(f"ERROR: Archivo de modelo no encontrado: {fnfe}")
            sys.exit(1)
        except DegirumException as de: # Usar excepción específica si es relevante
            logger.critical(f"ERROR: Fallo específico de Degirum/HailoRT al cargar el modelo '{self.model_name}': {de}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.critical(f"ERROR: No se pudo cargar el modelo '{self.model_name}': {e}", exc_info=True)
            sys.exit(1)
        
        return self.loaded_model
    

    def inference(self, frame):
        result = self.loaded_model(frame)
        raw_detections = getattr(result, 'results', None)
        if not raw_detections:
            raw_detections = getattr(result, '_inference_results', [])

        detections = []

        for det_dict in raw_detections:

            # 1️⃣ Comprobación y salto inmediato de objetos no válidos
            if not isinstance(det_dict, dict):
                logger.warning(f"Resultado no es dict: {type(det_dict)}")
                continue

            if not det_dict:
                # Dict vacío detectado
                logger.debug("Detección vacía (dict vacío), ignorando.")  # Usa DEBUG para no saturar con warnings
                continue

            # 2️⃣ Comprobación de claves obligatorias
            if 'bbox' not in det_dict or det_dict.get('score') is None:
                logger.warning(f"Detection object missing 'bbox' or 'score' key: {det_dict}")
                continue
        
            # 3️⃣ Si pasa todo, es una detección válida
            detections.append({
                'bbox':     det_dict['bbox'],
                'label':    det_dict.get('label', 'unknown'),
                'score':    det_dict['score'],
                'mask':     det_dict.get('mask', None)
            })

        if not detections:
            logger.info("No se detectaron objetos en el frame actual.")

        return detections

                
                