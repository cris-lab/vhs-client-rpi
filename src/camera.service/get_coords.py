import cv2
import numpy as np
import json
import sys
import os
import subprocess
import time
import signal

# --- Importar las funciones desde utils.py ---
try:
    from src.utils import resize_with_padding
except ImportError:
    print("Error: No se pudo importar 'resize_with_padding' de src/utils.py.")
    print("Asegúrate de que 'src' sea un paquete y 'utils.py' exista dentro de él.")
    print("Contenido mínimo para src/utils.py:")
    print("-----------------------------------")
    print("import cv2")
    print("import numpy as np")
    print("")
    print("def resize_with_padding(image, new_size):")
    print("    h, w = image.shape[:2]")
    print("    target_w, target_h = new_size")
    print("    aspect_ratio_img = w / h")
    print("    aspect_ratio_target = target_w / target_h")
    print("")
    print("    if aspect_ratio_img > aspect_ratio_target:")
    print("        # Imagen más ancha que alta, el ancho dominará el escalado")
    print("        new_w = target_w")
    print("        new_h = int(new_w / aspect_ratio_img)")
    print("    else:")
    print("        # Imagen más alta que ancha, el alto dominará el escalado")
    print("        new_h = target_h")
    print("        new_w = int(new_h * aspect_ratio_img)")
    print("")
    print("    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)")
    print("")
    print("    # Calcular padding")
    print("    pad_w = target_w - new_w")
    print("    pad_h = target_h - new_h")
    print("    top, bottom = pad_h // 2, pad_h - (pad_h // 2)")
    print("    left, right = pad_w // 2, pad_w - (pad_w // 2)")
    print("")
    print("    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))")
    print("    return padded_img")
    print("-----------------------------------")
    sys.exit(1)


# --- Configuración del archivo de configuración ---
FILE_PATH = '/var/lib/vhs' # O la ruta real donde está tu config.json
config_path = os.path.join(FILE_PATH, 'config.json')

# --- CONFIGURACIÓN: Tamaño fijo de la ventana de visualización (y para el modelo) ---
DISPLAY_WINDOW_SIZE = (640, 640) # Ancho y alto para la ventana de visualización
FIXED_SQUARE_SIZE = (640, 640) # Tamaño del cuadrado fijo que el usuario podría querer

# Lista para almacenar los puntos clicados (en coordenadas originales del frame grande)
points = []
# Color para dibujar los puntos y líneas (Verde)
drawing_color = (0, 255, 0)
# Color para el bounding box (Rojo)
bbox_color = (0, 0, 255)
# Color para el cuadrado fijo (Azul)
fixed_square_color = (255, 0, 0)

# --- Variables globales para el callback del ratón ---
current_display_image = None
original_image_shape = None # Guardará la forma (alto, ancho) de la imagen original antes de escalar para la UI
selection_mode = 'polygon' # 'polygon' para polígono/rectángulo, 'fixed_square' para cuadrado 640x640
fixed_square_top_left = None # Almacena la esquina superior izquierda del cuadrado fijo
original_display_image_for_reset = None # Variable global para resetear la imagen de visualización

def click_event(event, x, y, flags, param):
    """
    Función de callback para los eventos del ratón.
    Registra los clics izquierdos y dibuja los puntos.
    Los puntos se escalan de vuelta a las coordenadas originales (del frame grande)
    porque la imagen de visualización puede estar escalada o tener padding.
    """
    global points, current_display_image, original_image_shape, selection_mode, fixed_square_top_left, original_display_image_for_reset

    if event == cv2.EVENT_LBUTTONDOWN:
        h_disp_current, w_disp_current = current_display_image.shape[:2]

        orig_h, orig_w = original_image_shape

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_disp = w_disp_current / h_disp_current

        scaled_w, scaled_h = w_disp_current, h_disp_current

        if aspect_ratio_orig > aspect_ratio_disp:
            scaled_w = w_disp_current
            scaled_h = int(w_disp_current / aspect_ratio_orig)
        else:
            scaled_h = h_disp_current
            scaled_w = int(h_disp_current * aspect_ratio_orig)

        offset_x = (w_disp_current - scaled_w) // 2
        offset_y = (h_disp_current - scaled_h) // 2

        x_in_scaled_img = x - offset_x
        y_in_scaled_img = y - offset_y

        scale_factor_x = orig_w / scaled_w
        scale_factor_y = orig_h / scaled_h

        original_x = int(x_in_scaled_img * scale_factor_x)
        original_y = int(y_in_scaled_img * scale_factor_y)

        original_x = max(0, min(original_x, orig_w - 1))
        original_y = max(0, min(original_y, orig_h - 1))

        if selection_mode == 'polygon':
            points.append([original_x, original_y])
            print(f"Punto añadido (original): ({original_x}, {original_y})")
            print(f"Punto añadido (display): ({x}, {y})")

            cv2.circle(current_display_image, (x, y), 5, drawing_color, -1)

            if len(points) > 1:
                prev_original_x, prev_original_y = points[-2]

                inv_scale_factor_x = scaled_w / orig_w
                inv_scale_factor_y = scaled_h / orig_h

                prev_x_in_scaled_img = int(prev_original_x * inv_scale_factor_x)
                prev_y_in_scaled_img = int(prev_original_y * inv_scale_factor_y)

                prev_display_x = prev_x_in_scaled_img + offset_x
                prev_display_y = prev_y_in_scaled_img + offset_y

                cv2.line(current_display_image, (prev_display_x, prev_display_y), (x, y), drawing_color, 2)
        elif selection_mode == 'fixed_square':
            fixed_square_top_left = [original_x, original_y]
            print(f"Esquina superior izquierda del cuadrado {FIXED_SQUARE_SIZE[0]}x{FIXED_SQUARE_SIZE[1]} seleccionada (original): ({original_x}, {original_y})")
            print(f"Punto clicado (display): ({x}, {y})")

            temp_img = original_display_image_for_reset.copy()
            draw_fixed_square_on_display(temp_img, fixed_square_top_left, original_image_shape, DISPLAY_WINDOW_SIZE, fixed_square_color)
            current_display_image[:] = temp_img[:]


def draw_fixed_square_on_display(img_to_draw_on, top_left_orig, original_img_shape, display_window_size, color):
    """Dibuja el cuadrado fijo en la imagen de visualización."""
    if top_left_orig is None:
        return

    orig_h, orig_w = original_img_shape
    disp_w, disp_h = display_window_size[0], display_window_size[1]

    aspect_ratio_orig = orig_w / orig_h
    aspect_ratio_disp = disp_w / disp_h

    scaled_w, scaled_h = disp_w, disp_h
    if aspect_ratio_orig > aspect_ratio_disp:
        scaled_w = disp_w
        scaled_h = int(disp_w / aspect_ratio_orig)
    else:
        scaled_h = disp_h
        scaled_w = int(disp_h * aspect_ratio_orig)

    offset_x = (disp_w - scaled_w) // 2
    offset_y = (disp_h - scaled_h) // 2

    x_orig, y_orig = top_left_orig
    x_end_orig = min(x_orig + FIXED_SQUARE_SIZE[0], orig_w)
    y_end_orig = min(y_orig + FIXED_SQUARE_SIZE[1], orig_h)

    x_disp_start = int(x_orig * (scaled_w / orig_w)) + offset_x
    y_disp_start = int(y_orig * (scaled_h / orig_h)) + offset_y
    x_disp_end = int(x_end_orig * (scaled_w / orig_w)) + offset_x
    y_disp_end = int(y_end_orig * (scaled_h / orig_h)) + offset_y

    cv2.rectangle(img_to_draw_on, (x_disp_start, y_disp_start), (x_disp_end, y_disp_end), color, 2)


def get_roi_coordinates_from_image(image_path_or_frame, stream_name=""):
    """
    Permite al usuario seleccionar puntos en una imagen.
    image_path_or_frame: Ruta a una imagen o un array de numpy (frame).
    """
    global points, current_display_image, original_image_shape, selection_mode, fixed_square_top_left, original_display_image_for_reset
    points = []
    fixed_square_top_left = None
    selection_mode = 'polygon'

    if isinstance(image_path_or_frame, str):
        original_image = cv2.imread(image_path_or_frame)
        if original_image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path_or_frame}")
            return [], None, None
    else:
        original_image = image_path_or_frame.copy()

    if original_image is None:
        print("Error: Imagen no disponible para la selección de ROI.")
        return [], None, None

    original_image_shape = original_image.shape[:2]

    display_image = resize_with_padding(original_image, DISPLAY_WINDOW_SIZE)
    current_display_image = display_image.copy()
    original_display_image_for_reset = display_image.copy()

    window_name = f"Selecciona ROI para {stream_name} - Presiona 'c' para confirmar, 'r' para resetear, 'q' para salir, 's' para cuadrado 640x640"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, {})

    print(f"\n--- Modo de Selección de ROI para '{stream_name}' ---")
    print(f"Visualizando imagen redimensionada a {DISPLAY_WINDOW_SIZE[0]}x{DISPLAY_WINDOW_SIZE[1]} con padding.")
    print("\n--- Modo Actual: POLÍGONO / RECTÁNGULO ---")
    print("Haz clic izquierdo para añadir puntos:")
    print(" - Si seleccionas 2 puntos: se interpretará como las esquinas opuestas de un RECTÁNGULO.")
    print(" - Si seleccionas 3 o más puntos: se interpretará como un POLÍGONO libre.")
    print("\n--- Opciones de Teclas ---")
    print("Presiona 's' para cambiar al modo 'CUADRADO FIJO 640x640'.")
    print("Presiona 'c' para confirmar tu selección y ver los resultados.")
    print("Presiona 'r' para resetear y empezar de nuevo.")
    print("Presiona 'q' para salir sin guardar la selección actual.")
    print("--------------------------------------------------\n")

    while True:
        cv2.imshow(window_name, current_display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            fixed_square_top_left = None
            selection_mode = 'polygon'
            current_display_image = original_display_image_for_reset.copy()
            print("\n--- Puntos y modo reseteados. Vuelve a empezar en modo POLÍGONO / RECTÁNGULO. ---")
        elif key == ord('s'):
            points = []
            fixed_square_top_left = None
            selection_mode = 'fixed_square'
            current_display_image = original_display_image_for_reset.copy()
            print("\n--- Modo cambiado a: CUADRADO FIJO 640x640 ---")
            print("Haz un solo clic izquierdo para definir la esquina superior izquierda del cuadrado.")
            print("Presiona 'c' para confirmar o 'r' para resetear.")
        elif key == ord('c'):
            if selection_mode == 'polygon' and len(points) < 2:
                print("En modo POLÍGONO/RECTÁNGULO, necesitas al menos 2 puntos para definir un ROI.")
            elif selection_mode == 'fixed_square' and fixed_square_top_left is None:
                print("En modo CUADRADO FIJO, necesitas hacer un clic para definir la esquina superior izquierda.")
            else:
                break

    cv2.destroyAllWindows()

    final_points = []
    bbox_coords = None
    bbox_dimensions = None

    if selection_mode == 'fixed_square' and fixed_square_top_left is not None:
        x_orig, y_orig = fixed_square_top_left
        orig_h, orig_w = original_image_shape

        x_min = max(0, x_orig)
        y_min = max(0, y_orig)
        x_max = min(x_orig + FIXED_SQUARE_SIZE[0], orig_w)
        y_max = min(y_orig + FIXED_SQUARE_SIZE[1], orig_h)

        actual_width = x_max - x_min
        actual_height = y_max - y_min

        if actual_width < FIXED_SQUARE_SIZE[0] or actual_height < FIXED_SQUARE_SIZE[1]:
            print(f"Advertencia: El cuadrado de {FIXED_SQUARE_SIZE[0]}x{FIXED_SQUARE_SIZE[1]} se ha recortado para ajustarse a los límites de la imagen.")
            print(f"  Dimensiones finales: {actual_width}x{actual_height} píxeles.")
            if orig_w < FIXED_SQUARE_SIZE[0] or orig_h < FIXED_SQUARE_SIZE[1]:
                 print(f"  La imagen original ({orig_w}x{orig_h}) es más pequeña que el tamaño de cuadrado deseado.")
                 x_min, y_min = 0, 0
                 x_max, y_max = orig_w, orig_h
                 actual_width, actual_height = orig_w, orig_h

        final_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        bbox_coords = (x_min, y_min)
        bbox_dimensions = (actual_width, actual_height)

    elif selection_mode == 'polygon' and len(points) == 2:
        pt1_x, pt1_y = points[0]
        pt2_x, pt2_y = points[1]

        x_min = min(pt1_x, pt2_x)
        y_min = min(pt1_y, pt2_y)
        x_max = max(pt1_x, pt2_x)
        y_max = max(pt1_y, pt2_y)

        final_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        bbox_coords = (x_min, y_min)
        bbox_dimensions = (x_max - x_min + 1, y_max - y_min + 1)
    elif selection_mode == 'polygon' and len(points) >= 3:
        final_points = points
        points_np = np.array(points)
        x_min, y_min = np.min(points_np, axis=0)
        x_max, y_max = np.max(points_np, axis=0)
        bbox_coords = (x_min, y_min)
        bbox_dimensions = (x_max - x_min + 1, y_max - y_min + 1)
    else:
        print("No se seleccionaron suficientes puntos para definir un ROI.")
        return [], None, None

    if final_points:
        reconfirm_image = original_display_image_for_reset.copy()
        orig_h, orig_w = original_image_shape
        disp_w, disp_h = DISPLAY_WINDOW_SIZE

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_disp = disp_w / disp_h

        scaled_w, scaled_h = disp_w, disp_h
        if aspect_ratio_orig > aspect_ratio_disp:
            scaled_w = disp_w
            scaled_h = int(disp_w / aspect_ratio_orig)
        else:
            scaled_h = disp_h
            scaled_w = int(disp_h * aspect_ratio_orig)

        offset_x = (disp_w - scaled_w) // 2
        offset_y = (disp_h - scaled_h) // 2

        display_points = []
        for p_orig_x, p_orig_y in final_points:
            p_scaled_x = int(p_orig_x * (scaled_w / orig_w))
            p_scaled_y = int(p_orig_y * (scaled_h / orig_h))
            p_display_x = p_scaled_x + offset_x
            p_display_y = p_scaled_y + offset_y
            display_points.append([p_display_x, p_display_y])

        pts = np.array(display_points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(reconfirm_image, [pts], True, drawing_color, 2)

        if bbox_coords:
            bx_min, by_min = bbox_coords
            bx_max_draw = bx_min + bbox_dimensions[0] -1
            by_max_draw = by_min + bbox_dimensions[1] -1

            bbox_display_x_min = int(bx_min * (scaled_w / orig_w)) + offset_x
            bbox_display_y_min = int(by_min * (scaled_h / orig_h)) + offset_y
            bbox_display_x_max = int(bx_max_draw * (scaled_w / orig_w)) + offset_x
            bbox_display_y_max = int(by_max_draw * (scaled_h / orig_h)) + offset_y

            cv2.rectangle(reconfirm_image,
                          (bbox_display_x_min, bbox_display_y_min),
                          (bbox_display_x_max, bbox_display_y_max),
                          bbox_color, 2)

        cv2.imshow("ROI Final Confirmado", reconfirm_image)
        print("\nPresiona cualquier tecla para cerrar la ventana de confirmación.")
        cv2.waitKey(0)
        cv2.destroyWindow("ROI Final Confirmado")

    return final_points, bbox_coords, bbox_dimensions


if __name__ == "__main__":
    process = None # Inicializar 'process' aquí para el bloque finally
    orig_w = None # Inicializar estas variables aquí para que estén en el ámbito
    orig_h = None # y puedan ser accedidas en el bloque final de impresión

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Archivo de configuración no encontrado en {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: El archivo de configuración {config_path} no es un JSON válido.")
        sys.exit(1)

    if not config_data.get('streams'):
        print("Error: No se encontraron streams en el archivo de configuración.")
        sys.exit(1)

    target_stream = config_data['streams'][0]

    stream_url = target_stream['input']['url']
    stream_name = target_stream['name']

    temp_image_path = "/tmp/temp_frame_roi.jpg"
    original_captured_frame = None # Inicializa esta variable para el ámbito del try-except principal

    print(f"Intentando capturar un frame de '{stream_name}' ({stream_url}) usando gst-launch-1.0...")
    print(f"El frame se guardará temporalmente en: {temp_image_path}")

    # Es importante que el pipeline de gst-launch-1.0 genere un frame completo
    # para que cv2.imread pueda leer la resolución correctamente.
    gst_command_list = [
        "gst-launch-1.0",
        "-e", # Forzar EOS al finalizar
        f"rtspsrc", f"location={stream_url}", f"latency=100",
        "!", "queue", # Un pequeño buffer para estabilidad
        "!", "rtph264depay",
        "!", "h264parse", # Necesario para asegurar que el decodificador reciba un stream válido
        "!", "avdec_h264", # Decodificador de software H.264
        "!", "videoconvert", # Asegura la conversión a un formato compatible (ej. BGR para JPEG)
        "!", "jpegenc", # Codificador JPEG
        # CAMBIO AQUÍ: Elimina 'num-buffers=1'
        "!", f"filesink", f"location={temp_image_path}" # <-- ELIMINA '", "num-buffers=1"' de aquí
    ]

    try:
        print("Iniciando proceso GStreamer para capturar 1 frame...")
        # Usamos preexec_fn=os.setsid para crear un nuevo grupo de procesos,
        # lo que nos permite matar todo el grupo si el proceso principal cuelga.
        process = subprocess.Popen(gst_command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)

        timeout_seconds = 10 # Aumentar el timeout para dar más tiempo a la captura
        start_time = time.time()
        frame_captured = False

        while time.time() - start_time < timeout_seconds:
            # Check for a non-empty file to ensure a valid frame was captured
            if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 1000:
                print(f"Frame capturado en {temp_image_path} (Tamaño: {os.path.getsize(temp_image_path)} bytes). Terminando proceso GStreamer...")
                frame_captured = True
                break
            if process.poll() is not None: # Si el proceso GStreamer ha terminado
                print("El proceso GStreamer terminó inesperadamente antes de capturar el frame.")
                break
            time.sleep(0.5)

        if not frame_captured:
            print(f"No se pudo capturar un frame válido en {timeout_seconds} segundos. El archivo de imagen temporal no se creó o está vacío.")
            print("Asegúrate de que la URL del stream sea correcta y accesible, y que GStreamer pueda decodificarlo.")
        else:
            print("Terminando proceso GStreamer.")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM) # Enviar SIGTERM al grupo
            try:
                process.wait(timeout=2) # Esperar un poco a que termine limpiamente
            except subprocess.TimeoutExpired:
                print("GStreamer no terminó a tiempo, forzando la terminación.")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL) # Forzar terminación con SIGKILL

        # Capturar stdout/stderr solo si el proceso ha terminado
        stdout, stderr = process.communicate()
        if stdout:
            print("Salida de GStreamer (stdout):\n", stdout)
        if stderr:
            print("Salida de GStreamer (stderr):\n", stderr)

        # --- AQUÍ EMPIEZAN LOS CAMBIOS PARA MOSTRAR LA RESOLUCIÓN ---
        if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
            original_captured_frame = cv2.imread(temp_image_path)
            if original_captured_frame is not None:
                orig_h, orig_w = original_captured_frame.shape[:2] # <<-- Obtener alto y ancho aquí
                print(f"\n--- Resolución Original del Frame Capturado: {orig_w}x{orig_h} píxeles ---")
                
                # Pasar la imagen original al selector de ROI
                selected_points, bbox_coords, bbox_dimensions = get_roi_coordinates_from_image(original_captured_frame, stream_name)
            else:
                print(f"\nAdvertencia: No se pudo leer la imagen capturada en {temp_image_path} para determinar su resolución o para la selección de ROI.")
                selected_points = []
                bbox_coords = None
                bbox_dimensions = None
        else:
            selected_points = []
            bbox_coords = None
            bbox_dimensions = None
        # --- AQUÍ TERMINAN LOS CAMBIOS EN ESTE BLOQUE ---

    except FileNotFoundError:
        print("Error: 'gst-launch-1.0' no encontrado. Asegúrate de que GStreamer está instalado y en tu PATH.")
        selected_points = []
        bbox_coords = None
        bbox_dimensions = None
    except Exception as e:
        print(f"Error inesperado durante la captura del frame: {e}")
        selected_points = []
        bbox_coords = None
        bbox_dimensions = None
    finally:
        # Asegurarse de que el proceso GStreamer se detenga si aún está corriendo
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                pass # El proceso ya murió
        # Eliminar el archivo temporal
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"Archivo temporal '{temp_image_path}' eliminado.")

    print("\n--- Coordenadas Finales Seleccionadas ---")
    if selected_points:
        print(f"Coordenadas para '{stream_name}':")
        print(selected_points)
        if bbox_coords and bbox_dimensions:
            print(f"\nRectángulo Delimitador (Bounding Box) en el frame original:")
            print(f"  Esquina Superior Izquierda: ({bbox_coords[0]}, {bbox_coords[1]})")
            print(f"  Dimensiones (Ancho x Alto): {bbox_dimensions[0]} x {bbox_dimensions[1]} píxeles")

        print("\nCopia estas coordenadas y pégalas en la sección 'zones' de tu archivo de configuración (config.json) para este stream.")
        print("Ten en cuenta que:")
        if selection_mode == 'fixed_square':
            print(f" - Seleccionaste un CUADRADO FIJO de {FIXED_SQUARE_SIZE[0]}x{FIXED_SQUARE_SIZE[1]} (o lo más cercano posible si la imagen es más pequeña).")
            # --- AQUÍ SE REPITE LA IMPRESIÓN DE LA RESOLUCIÓN ORIGINAL PARA CONSISTENCIA ---
            if original_captured_frame is not None:
                orig_h_final, orig_w_final = original_captured_frame.shape[:2]
                print(f" - La resolución original de la fuente es {orig_w_final}x{orig_h_final} píxeles.")
            # --- FIN DE LA REPETICIÓN ---
        else:
            print(" - Si seleccionaste 2 puntos, estas coordenadas forman un RECTÁNGULO.")
            print(" - Si seleccionaste 3 o más puntos, estas coordenadas forman un POLÍGONO.")
        print("El 'Rectángulo Delimitador' te da las dimensiones generales que abarca tu selección en el frame original.")
    else:
        print("No se seleccionaron puntos o hubo un error al capturar el frame.")