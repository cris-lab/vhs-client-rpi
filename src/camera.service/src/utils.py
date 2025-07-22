import json
import sys
import numpy as np
import cv2
import io
from PIL import Image

def crop_and_resize_roi_padded(frame, roi={'enabled':False, 'points': []}, target_size=(640, 640), debug=True):
    
    if debug:
        print(f"\n--- DEBUG: Inicia crop_and_resize_roi_padded ---")
        print(f"DEBUG: Frame de entrada shape: {frame.shape}")
        print(f"DEBUG: ROI: {roi}")
    
    # Cambié [] por None como valor predeterminado, y luego verifico si es una lista vacía.
    # Es más robusto en caso de que se pase None explícitamente.
    if roi is None or len(roi['points']) == 0:
        if debug:
            print("DEBUG: No hay ROI definida (o lista vacía). Procesando frame completo.")
        result_frame = resize_with_padding(frame, target_size)
        
        if debug:
            print(f"DEBUG: Final output shape (No ROI): {result_frame.shape}")
        return result_frame

    # Convertir puntos de ROI a formato adecuado para OpenCV
    roi_polygon = np.array(roi['points'], dtype=np.int32)

    # Obtener el bounding box del ROI
    x, y, w, h = cv2.boundingRect(roi_polygon)
    
    if debug:
        print(f"DEBUG: ROI Bounding Box (x, y, w, h): {x}, {y}, {w}, {h}")

    # Asegurarse de que el bounding box sea válido (no 0 ancho/alto)
    if w == 0 or h == 0:
        
        if debug:
            print("Advertencia: Bounding box de ROI tiene ancho o alto de cero. Retornando frame completo redimensionado.")
        result_frame = resize_with_padding(frame, target_size)
        
        if debug:
            print(f"DEBUG: Final output shape (ROI invalid): {result_frame.shape}")
            
        return result_frame

    # Recortar el ROI del frame original
    # Asegúrate de que los índices no se salgan de los límites del frame
    # Estos 'max(0,...)' y 'min(frame.shape[X],...)' son para evitar errores si el BB se sale de la imagen
    cropped_frame = frame[max(0, y):min(frame.shape[0], y+h), \
                          max(0, x):min(frame.shape[1], x+w)]
    
    if debug:
        print(f"DEBUG: Cropped frame shape: {cropped_frame.shape}")

    # Si el cropped_frame resultante está vacío después del recorte (por ejemplo, si el BB estaba fuera de la imagen)
    if cropped_frame.size == 0:
        
        if debug:
            print("Advertencia: El recorte del ROI resultó en un frame vacío. Retornando frame completo redimensionado.")
        result_frame = resize_with_padding(frame, target_size)
        
        if debug:
            print(f"DEBUG: Final output shape (Cropped empty): {result_frame.shape}")
        return result_frame

    # Redimensionar el recorte a 640x640 manteniendo la proporcionalidad
    resized_padded_frame = resize_with_padding(cropped_frame, target_size)
    
    if debug:
        print(f"DEBUG: Final output shape (with ROI): {resized_padded_frame.shape}")

    return resized_padded_frame

def resize_with_padding(image, target_size, debug=False):
    """
    Redimensiona una imagen al target_size (ancho, alto) manteniendo la relación de aspecto
    y añadiendo relleno (padding) negro si es necesario.
    """
    
    if debug:
        print(f"--- DEBUG: Inicia resize_with_padding ---")
        print(f"DEBUG: Image received by resize_with_padding shape: {image.shape}")
        print(f"DEBUG: Target size: {target_size}")

    h_orig, w_orig = image.shape[:2]
    target_w, target_h = target_size

    # Manejar el caso de imágenes vacías o con dimensiones cero
    if w_orig == 0 or h_orig == 0:
        
        if debug:
            print(f"Advertencia: Imagen original con dimensiones cero ({w_orig}x{h_orig}) en resize_with_padding. Retornando imagen negra del target_size.")
        return np.full((target_h, target_w, 3), 0, dtype=np.uint8) # Asumo 3 canales para compatibilidad

    # Calcular la relación de aspecto de la imagen original y la del destino
    aspect_ratio_orig = w_orig / h_orig
    aspect_ratio_target = target_w / target_h
    
    if debug:
        print(f"DEBUG: Aspect Ratios - Original: {aspect_ratio_orig:.2f}, Target: {aspect_ratio_target:.2f}")

    new_w, new_h = target_w, target_h # Valores por defecto

    if aspect_ratio_orig > aspect_ratio_target:
        # La imagen original es más ancha que el destino (o el destino es más alto)
        new_w = target_w
        new_h = int(target_w / aspect_ratio_orig)
        
        if debug:
            print(f"DEBUG: Aspect ratio condition: Original is wider.")
    else:
        # La imagen original es más alta que el destino (o el destino es más ancho)
        new_h = target_h
        new_w = int(target_h * aspect_ratio_orig)
        
        if debug:
            print(f"DEBUG: Aspect ratio condition: Original is taller or same.")

    # Asegurarse de que las nuevas dimensiones sean al menos 1 para evitar errores de cv2.resize
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    if debug:
        print(f"DEBUG: Calculated proportional dimensions (w, h): {new_w}, {new_h}")

    # Redimensionar la imagen manteniendo la proporcionalidad
    try:
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if debug:
            print(f"DEBUG: Resized image shape from cv2.resize: {resized_image.shape}")
            
    except Exception as e:
        print(f"ERROR: Fallo al redimensionar la imagen con cv2.resize: {e}")
        # En caso de error crítico al redimensionar, retornar una imagen negra
        return np.full((target_h, target_w, 3), 0, dtype=np.uint8)

    # Crear una nueva imagen del tamaño objetivo con fondo negro
    # Determinar el número de canales de la imagen de entrada para que el padding sea compatible
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    padded_image = np.full((target_h, target_w, num_channels), 0, dtype=np.uint8)
    
    if debug:
        print(f"DEBUG: Padded canvas image shape: {padded_image.shape}")

    # Calcular la posición para pegar la imagen redimensionada centrada
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    
    if debug:
        print(f"DEBUG: Padding start coordinates (x, y): {start_x}, {start_y}")

    end_y = start_y + new_h
    end_x = start_x + new_w
    
    if debug:
        print(f"DEBUG: Padding end coordinates (x, y): {end_x}, {end_y}")
        print(f"DEBUG: Target slice dimensions (h, w): {(end_y - start_y)}, {(end_x - start_x)}")

    # Asegurarse de que el slice de destino y la imagen redimensionada tengan las mismas dimensiones
    if resized_image.shape[0] != (end_y - start_y) or resized_image.shape[1] != (end_x - start_x):
        
        if debug:
            print(f"ERROR: Desajuste de dimensiones en la asignación final.")
            print(f"  resized_image_shape={resized_image.shape}")
            print(f"  target_slice_shape={(end_y - start_y, end_x - start_x)}")
        # Este es un fallback. Debería indicar un problema lógico si se llega aquí.
        # Intenta un resize final directo al target_size si todo lo demás falla.
        try:
            final_fallback_resize = cv2.resize(resized_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            if debug:
                print("DEBUG: Usando fallback resize directo al target_size.")
            return final_fallback_resize
        except Exception as e:
            print(f"ERROR: Fallback resize también falló: {e}. Retornando imagen negra.")
            return np.full((target_h, target_w, 3), 0, dtype=np.uint8) # Fallback a imagen negra

    # Pegar la imagen redimensionada en el centro de la imagen con padding
    padded_image[start_y:end_y, start_x:end_x] = resized_image
    
    if debug:
        print(f"--- DEBUG: Fin resize_with_padding ---")
        
    return padded_image



def _preprocess_image_for_model(image_data: bytes, target_width: int, target_height: int):
    """
    Preprocesa la imagen para adaptarla a las dimensiones.
    Siempre devuelve un array NumPy de tipo UINT8 (0-255).
    """
    image = Image.open(io.BytesIO(image_data)).convert("RGB") # Asegurar 3 canales RGB
    ancho_original, alto_original = image.size
    
    # Calcular escala para mantener la relación de aspecto y ajustar a las dimensiones objetivo
    escala = min(target_width / ancho_original, target_height / alto_original)
    nuevo_ancho = int(ancho_original * escala)
    nuevo_alto = int(alto_original * escala)
    
    # Redimensionar imagen
    image_resized = image.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
    
    # Crear un lienzo (fondo) del tamaño objetivo y pegar la imagen redimensionada
    # El color gris neutro (128, 128, 128) es común para el padding
    lienzo = Image.new("RGB", (target_width, target_height), (128, 128, 128))
    pad_x = (target_width - nuevo_ancho) // 2
    pad_y = (target_height - nuevo_alto) // 2
    lienzo.paste(image_resized, (pad_x, pad_y))
    
    # Devolver el array NumPy como UINT8.
    image_listo_para_modelo = np.array(lienzo, dtype=np.uint8) 

    return image_listo_para_modelo, pad_x, pad_y, escala, ancho_original, alto_original


def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"ERROR: El archivo de configuración no se encontró en '{config_path}'.")
        print("Por favor, asegúrese de que 'config.json' exista en la ruta especificada.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: El archivo '{config_path}' no es un JSON válido.")
        print("Por favor, revise la sintaxis del archivo de configuración.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ocurrió un error inesperado al cargar la configuración: {e}")
        sys.exit(1)



def draw_tracks(roi, tracker_results, person_data):
    """
    Dibuja la máscara si está disponible; si no, el bounding box.
    """
    for track in tracker_results:
        x, y, w, h = map(int, track.tlwh)
        mask = getattr(track, 'mask', None)
        track_id = track.track_id

        pdata = person_data.get(track_id)
        origin = pdata.get("origin_id", track_id) if pdata else track_id

        color = (0, 255, 0) if track_id == origin else (0, 128, 255)

        # --- Dibujar máscara si está disponible ---
        if mask is not None and mask.shape == roi.shape[:2]:
            mask_colored = np.zeros_like(roi, dtype=np.uint8)
            mask_colored[mask == 1] = color
            cv2.addWeighted(mask_colored, 0.4, roi, 0.6, 0, roi)
        else:
            # Si no hay máscara, dibujar bounding box
            cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)

        # --- Etiqueta con metadatos ---
        label = f"ID:{origin}"
        if pdata:
            if pdata.get("gender"):
                label += f" {pdata['gender']}"
            if pdata.get("age"):
                label += f" {pdata['age']}"
            if pdata.get("description"):
                label += f" | {pdata['description'][:18]}..."
            movement_state = pdata.get("movement_state", "Desconocido")
            label += f" | {movement_state}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(roi, (x, y - th - 5), (x + tw, y), color, -1)
        cv2.putText(roi, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return roi

def draw_grid_on_frame(frame, grid_size=6, color=(0, 255, 0), thickness=1):
    """
    Dibuja una grilla en el frame según el tamaño especificado.
    """
    height, width, _ = frame.shape
    cell_width = width // grid_size
    cell_height = height // grid_size

    # Dibujar líneas verticales
    for i in range(1, grid_size):
        x = i * cell_width
        cv2.line(frame, (x, 0), (x, height), color, thickness)

    # Dibujar líneas horizontales
    for i in range(1, grid_size):
        y = i * cell_height
        cv2.line(frame, (0, y), (width, y), color, thickness)

    # (Opcional) Dibujar texto en cada celda
    for row in range(grid_size):
        for col in range(grid_size):
            zone_label = f"Z{row}{col}"
            text_x = col * cell_width + 5
            text_y = row * cell_height + 20
            cv2.putText(frame, zone_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
