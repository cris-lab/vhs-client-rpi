 
import cv2, uuid, degirum_tools, numpy as np, time
from src.ModelLoader import ModelLoader
from typing import Tuple, Dict, Any


def check_line_crossing(point: Tuple[float, float], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    position = (y2 - y1) * x + (x1 - x2) * y + (x2 * y1 - x1 * y2)
    return position > 0

def get_bbox_anchor_point(bbox: Tuple[int, int, int, int], anchor_type: degirum_tools.AnchorPoint) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    if anchor_type == degirum_tools.AnchorPoint.CENTER:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    elif anchor_type == degirum_tools.AnchorPoint.BOTTOM_CENTER:
        return ((x1 + x2) / 2, y2)
    elif anchor_type == degirum_tools.AnchorPoint.TOP_CENTER:
        return ((x1 + x2) / 2, y1)
    elif anchor_type == degirum_tools.AnchorPoint.BOTTOM_LEFT:
        return (x1, y2)
    elif anchor_type == degirum_tools.AnchorPoint.BOTTOM_RIGHT:
        return (x2, y2)
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# --- Clase CustomLineCounter ---
class CustomLineCounter:
    def __init__(
        self,
        line: Tuple[int, int, int, int],
        count_direction: str = "HORIZONTAL",
        anchor_point: degirum_tools.AnchorPoint = degirum_tools.AnchorPoint.BOTTOM_CENTER,
        *,
        count_first_crossing: bool = True,
        annotation_color: Tuple[int, int, int] = (255, 255, 0), # Color de la línea y fondo del texto
        annotation_text_color: Tuple[int, int, int] = (255, 255, 255), # Color del texto (blanco)
        annotation_line_width: int = 2, # Grosor de la línea de conteo
        annotation_font_scale: float = 0.4,
        annotation_text_thickness: int = 1, # NUEVO: Grosor del texto (1 para delgado)
        annotation_text_margin: int = 10,
        name: str = "Linea",
        class_list: list = [],
        on_cross_callbacks=None
    ):
        self.line = np.array(line).astype(int)
        self.count_direction = count_direction
        self.count_first_crossing = count_first_crossing
        self.annotation_color = annotation_color
        self.annotation_text_color = annotation_text_color
        self.annotation_line_width = annotation_line_width
        self.annotation_font_scale = annotation_font_scale
        self.annotation_text_thickness = annotation_text_thickness # Guardar el nuevo grosor del texto
        self.annotation_text_margin = annotation_text_margin
        self.name = name
        self.class_list = class_list
        self.entry_count = 0
        self.exit_count = 0
        self._last_side: Dict[int, bool] = {}
        self._counted_trails: Dict[int, bool] = {}
        self.anchor_point = anchor_point
        self.on_cross_callbacks = on_cross_callbacks or []
        

    def _intersect_segments(self, p1: Tuple[float, float], q1: Tuple[float, float], p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - \
                  (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True

        return False

    def analyze(self, result: Any):
        if not hasattr(result, "trails") or not result.trails:
            self._last_side = {}
            self._counted_trails = {}
            return

        # Construir diccionario: track_id -> class_name
        tid_to_class = {}
        if hasattr(result, "results"):
            for det in result.results:
                tid = det.get("track_id")
                class_name = det.get("class_name") or det.get("label") or det.get("class")
                if tid is not None and class_name is not None:
                    tid_to_class[tid] = class_name

        line_p1 = tuple(self.line[:2])
        line_p2 = tuple(self.line[2:])

        for tid, trail_bboxes in result.trails.items():
            # 🔍 Filtro por clase
            if self.class_list:
                class_name = tid_to_class.get(tid)
                if class_name not in self.class_list:
                    continue

            if not trail_bboxes or len(trail_bboxes) < 2:
                if trail_bboxes and tid not in self._last_side:
                    current_bbox = trail_bboxes[-1]
                    current_point = get_bbox_anchor_point(current_bbox, self.anchor_point)
                    self._last_side[tid] = check_line_crossing(current_point, line_p1, line_p2)
                continue

            current_bbox = trail_bboxes[-1]
            last_bbox = trail_bboxes[-2]

            current_point = get_bbox_anchor_point(current_bbox, self.anchor_point)
            last_point = get_bbox_anchor_point(last_bbox, self.anchor_point)

            current_side = check_line_crossing(current_point, line_p1, line_p2)

            if tid not in self._last_side:
                self._last_side[tid] = current_side
                continue

            segments_intersect = self._intersect_segments(line_p1, line_p2, last_point, current_point)

            if (
                self._last_side[tid] != current_side
                and not self._counted_trails.get(tid, False)
                and segments_intersect
            ):
                sentido = (self._last_side[tid], current_side)
                direction = 'Unknown';
                
                if self.count_direction == "HORIZONTAL":
                    if sentido == (False, True):
                        direction = "Right"
                        self.entry_count += 1
                    elif sentido == (True, False):
                        direction = "Left"
                        self.exit_count += 1
                elif self.count_direction == "VERTICAL":
                    if sentido == (False, True):
                        direction = "Down"
                        self.entry_count += 1
                    elif sentido == (True, False):
                        direction = "Up"
                        self.exit_count += 1
                else:
                    self.entry_count += 1
                
                event = {
                    "tid": tid,
                    "uuid": str(uuid.uuid4()),
                    "name": self.name,
                    "type": "person_crossed_line",
                    "direction": direction,
                    "class_name": tid_to_class.get(tid, "Unknown"),
                    "timestamp": time.time(),
                }
                
                for callback in self.on_cross_callbacks:
                    callback(event)

                print(f"🟢 {self.name} TID {tid} CRUCE DETECTADO! Entrada: {self.entry_count}, Salida: {self.exit_count}")
                self._counted_trails[tid] = True if self.count_first_crossing else False

            self._last_side[tid] = current_side

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja la línea de conteo y los conteos en el fotograma con fondo y texto blanco.
        Posiciona las etiquetas de conteo alineadas.
        """
        # Dibuja la línea
        cv2.line(frame, tuple(self.line[:2]), tuple(self.line[2:]), self.annotation_color, self.annotation_line_width)

        x1, y1, x2, y2 = self.line
        is_vertical = abs(y2 - y1) > abs(x2 - x1)

        # Configuración para el fondo del texto
        text_bg_padding = 5
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # Función auxiliar para dibujar texto con fondo
        def draw_text_with_background(img, text, org, font, font_scale, text_color, bg_color, thickness, padding):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            x, y = org[0], org[1]
            top_left = (x - padding, y - text_height - baseline - padding)
            bottom_right = (x + text_width + padding, y + padding)
            
            top_left = (max(0, top_left[0]), max(0, top_left[1]))
            bottom_right = (min(img.shape[1], bottom_right[0]), min(img.shape[0], bottom_right[1]))

            cv2.rectangle(img, top_left, bottom_right, bg_color, cv2.FILLED)
            
            # Aquí es donde aplicamos el grosor de texto y el antialiasing
            cv2.putText(img, text, (x, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)
            return img


        if is_vertical:
            
            label_a = "Left"
            label_b = "Right"

            text_a_str = f"{label_a}: {self.entry_count}"
            text_b_str = f"{label_b}: {self.exit_count}"

            # Es importante usar el grosor del texto para el cálculo del tamaño del texto
            (text_a_width, text_a_height), baseline_a = cv2.getTextSize(
                text_a_str, font_face, self.annotation_font_scale, self.annotation_text_thickness
            )
            (text_b_width, text_b_height), baseline_b = cv2.getTextSize(
                text_b_str, font_face, self.annotation_font_scale, self.annotation_text_thickness
            )

            # Posición Izquierda
            pos_a_x = min(x1, x2) - text_a_width - self.annotation_text_margin - text_bg_padding
            pos_a_y = (y1 + y2) // 2 + text_a_height // 2 # Posición Y para la base del texto
            
            pos_a_x = max(text_bg_padding, pos_a_x)
            pos_a_y = min(frame.shape[0] - text_bg_padding, pos_a_y)
            pos_a_y = max(text_a_height + baseline_a + text_bg_padding, pos_a_y)

            # Posición Derecha
            pos_b_x = max(x1, x2) + self.annotation_text_margin + text_bg_padding
            pos_b_y = (y1 + y2) // 2 + text_b_height // 2

            pos_b_x = min(frame.shape[1] - text_b_width - text_bg_padding, pos_b_x)
            pos_b_y = min(frame.shape[0] - text_bg_padding, pos_b_y)
            pos_b_y = max(text_b_height + baseline_b + text_bg_padding, pos_b_y)

            draw_text_with_background(
                frame, text_a_str, (pos_a_x, pos_a_y),
                font_face, self.annotation_font_scale, self.annotation_text_color,
                self.annotation_color, self.annotation_text_thickness, text_bg_padding
            )

            draw_text_with_background(
                frame, text_b_str, (pos_b_x, pos_b_y),
                font_face, self.annotation_font_scale, self.annotation_text_color,
                self.annotation_color, self.annotation_text_thickness, text_bg_padding
            )

        else:
            
            label_a = "Up"
            label_b = "Down"

            text_a_str = f"{label_a}: {self.exit_count}"
            text_b_str = f"{label_b}: {self.entry_count}"

            # Es importante usar el grosor del texto para el cálculo del tamaño del texto
            (text_a_width, text_a_height), baseline_a = cv2.getTextSize(
                text_a_str, font_face, self.annotation_font_scale, self.annotation_text_thickness
            )
            (text_b_width, text_b_height), baseline_b = cv2.getTextSize(
                text_b_str, font_face, self.annotation_font_scale, self.annotation_text_thickness
            )

            # Posición Arriba
            pos_a_x = min(x1, x2) + self.annotation_text_margin
            pos_a_y = min(y1, y2) - self.annotation_text_margin
            
            pos_a_x = max(text_bg_padding, pos_a_x)
            pos_a_y = max(text_a_height + baseline_a + text_bg_padding, pos_a_y)

            # Posición Abajo
            pos_b_x = min(x1, x2) + self.annotation_text_margin
            pos_b_y = max(y1, y2) + text_b_height + self.annotation_text_margin + text_bg_padding

            pos_b_x = max(text_bg_padding, pos_b_x)
            pos_b_y = min(frame.shape[0] - text_bg_padding, pos_b_y)

            draw_text_with_background(
                frame, text_a_str, (pos_a_x, pos_a_y),
                font_face, self.annotation_font_scale, self.annotation_text_color,
                self.annotation_color, self.annotation_text_thickness, text_bg_padding
            )

            draw_text_with_background(
                frame, text_b_str, (pos_b_x, pos_b_y),
                font_face, self.annotation_font_scale, self.annotation_text_color,
                self.annotation_color, self.annotation_text_thickness, text_bg_padding
            )
        
        return frame