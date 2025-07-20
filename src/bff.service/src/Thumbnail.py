import os
import time
import cv2

def _maybe_save_thumbnail(frame, stream_id, last_time, interval, base_path):
    """Guarda un thumbnail JPEG del frame si ha pasado el tiempo suficiente."""
    try:
        current_time = time.time()

        if frame is None or frame.size == 0:
            print("[✗] Frame vacío, no se puede guardar thumbnail.")
            return last_time

        if current_time - last_time >= interval:
            os.makedirs(base_path, exist_ok=True)
            thumbnail_path = os.path.join(base_path, f"thumbnail_{stream_id}.jpg")
            success = cv2.imwrite(thumbnail_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if success:
                print(f"[✓] Thumbnail guardado en: {thumbnail_path}")
                return current_time
            else:
                print("[✗] Falló al guardar el thumbnail.")
    except Exception as e:
        print(f"[✗] Error al guardar thumbnail: {e}")
    return last_time


def _get_thumbnail_path(stream_id, base_path):
    """Devuelve la ruta del thumbnail para un stream específico."""
    return os.path.join(base_path, f"thumbnail_{stream_id}.jpg")