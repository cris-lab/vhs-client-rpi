import numpy as np
import cv2
import degirum_tools
from src.ModelLoader import ModelLoader
from src.CustomLineCounter import CustomLineCounter
from src.EventProcessor import EventProcessor
from typing import Tuple, Dict, Any
from src.HeatMap import HeatMap

class FrameProcessor:
    
    def __init__(self, config: Dict[str, Any], stream: Dict[str, Any]):
        self.stream = stream

        self.tracker = degirum_tools.ObjectTracker(
            class_list=['head', 'person'],
            track_thresh=stream.get('tracker', {}).get('track_thresh', 0.5),
            track_buffer=stream.get('tracker', {}).get('track_buffer', 30),
            match_thresh=stream.get('tracker', {}).get('match_thresh', 20),
            trail_depth=stream.get('tracker', {}).get('trail_depth', 20),
            anchor_point=degirum_tools.AnchorPoint.CENTER,
            annotation_color=(255, 0, 0),
        )
        
        self.event_processor = EventProcessor(config, stream)
        self.line_counters = self.create_counters(stream)
        
        self.combined_model = degirum_tools.CombiningCompoundModel(
            ModelLoader('yolo11n_silu_coco--640x640_quant_hailort_hailo8l_1').load_model(),
            ModelLoader('yolov8n_relu6_face--640x640_quant_hailort_hailo8l_1').load_model()
        )
        
        self.heatmap = HeatMap(
            model=self.combined_model,
            frame_size=(640, 640),
            grid_size=(20, 20),
            decay_factor=0.9
        )

        print("[✔] FrameProcessor inicializado")

    def execute(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        
        heat_map    = self.heatmap.analyze(frame)
        result      = self.combined_model(frame)
        
        self.filtrar_detecciones_validas(result.results)

        if len(result.results) > 0:
            current_track_ids_in_result = set(
                r['track_id'] for r in result.results if 'track_id' in r
            )

            to_remove = [
                e for e in self.event_processor.event_tracker
                if e['tid'] not in current_track_ids_in_result and not e.get('inference_in_progress', False)
            ]
            if to_remove:
                print(f"[🧹] Posibles eventos a eliminar: {[e['tid'] for e in to_remove]}")

            self.tracker.analyze(result)

            for counter in self.line_counters:
                counter.analyze(result)

            self.event_processor.analyze(result, frame)

            frame = self.tracker.annotate(result, frame)

            for res in result.results:
                label = res.get('label', '').lower()
                if label == 'human face':
                    x1, y1, x2, y2 = map(int, res['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    track_id = res.get('track_id')
                    if track_id is not None:
                        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for counter in self.line_counters:
            frame = counter.annotate(frame)
        
        return frame, True

    def filtrar_detecciones_validas(self, result_list: list):
        indices_a_eliminar = []
        for idx, detection in enumerate(result_list):
            if 'bbox' not in detection or 'label' not in detection:
                indices_a_eliminar.append(idx)
        for idx in reversed(indices_a_eliminar):
            del result_list[idx]
        if indices_a_eliminar:
            print(f"[⚠️] Detecciones inválidas eliminadas: {indices_a_eliminar}")

    def create_counters(self, stream):
        counters = []
        for element in stream.get('lines'):
            callbacks = []
            for cb_name in element.get("call_backs", []):
                cb_func = getattr(self.event_processor, cb_name, None)
                if cb_func:
                    callbacks.append(cb_func)
            counter = CustomLineCounter(
                line=element.get('points'),
                count_direction=element.get('direction').upper(),
                anchor_point=degirum_tools.AnchorPoint.CENTER,
                name=element.get('name'),
                class_list=element.get('class', []),
                on_cross_callbacks=callbacks,
                annotation_color=(255, 0, 0),
                annotation_line_width=2,
                annotation_text_margin=5,
                annotation_text_thickness=1,
            )
            counters.append(counter)
        print(f"[✔] Line counters creados: {len(counters)}")
        return counters
