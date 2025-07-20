
import degirum_tools, cv2
from src.ModelLoader import ModelLoader

class FaceFeatures:
    
    model = degirum_tools.CombiningCompoundModel(
        degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8s_relu6_peta_pedestrian_attributes--128x256_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1').load_model(),
        ),
        degirum_tools.CombiningCompoundModel(
            ModelLoader('yolov8n_relu6_age--256x256_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_imdbage_bmse--224x224_quant_hailort_multidevice_2').load_model(),
        )
    )
    
    def execute(self, frame):
        """
        Ejecuta el análisis de detecciones faciales en un frame.
        Retorna el frame procesado y un booleano indicando si se detectaron rostros.
        """
        
