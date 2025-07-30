import argparse
import os
import time
import datetime
import cv2
import src.utils as vhs_utils
from src.FrameProcessor import FrameProcessor
from degirum_tools.video_support import (
    open_video_stream,
    video_source,
)

parser = argparse.ArgumentParser(description='VHS Camera Stream Service')
parser.add_argument('--stream-id', type=str, required=False, help='ID del stream a procesar')

args = parser.parse_args()
stream_id = args.stream_id

config = vhs_utils.load_config(os.path.join('/var/lib/vhs', 'config.json'))

if stream_id:
    stream_config = next((s for s in config.get('streams', []) if s['id'] == stream_id), None)
else:
    stream_config = config.get('streams', [])[0]

if stream_config is None:
    raise ValueError(f"No se encontró ningún stream con el ID: {stream_id}")

video_source_url = stream_config['input']['url']
frame_processor = FrameProcessor(config, stream_config)


def video_source_buffered(stream, fps=30.0, buffer_size=30):
    q = queue.Queue(maxsize=buffer_size)
    stop_flag = threading.Event()

    def capture():
        for frame in video_source(stream, fps=fps):
            if stop_flag.is_set():
                break
            try:
                q.put(frame, timeout=0.1)
            except queue.Full:
                pass  # descarta si está lleno para no bloquear lectura

    threading.Thread(target=capture, daemon=True).start()

    try:
        while not stop_flag.is_set():
            try:
                frame = q.get(timeout=1.0)
                yield frame
            except queue.Empty:
                break
    finally:
        stop_flag.set()

# --- Main Script ---
if __name__ == "__main__":
    print(f"Intentando abrir el stream RTSP: {video_source_url}")

    try:
        with open_video_stream(video_source_url) as stream:
            print("Stream de video abierto exitosamente.")

            width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = stream.get(cv2.CAP_PROP_FPS)
            print(f"Dimensiones del stream: {width}x{height}, FPS: {fps:.2f}")

            start_time  = time.time()
            frame_count = 0
            frame_skip  = 1  

            for frame in video_source(stream, fps=30.0):

                # --- Grabar frame sin procesar ---
                current_time = time.time()
                # Procesamiento posterior (ROI, etc.)
                roi_frame = vhs_utils.crop_and_resize_roi_padded(frame, stream_config.get('roi', None), target_size=(640, 640))
                processed_frame, _ = frame_processor.execute(roi_frame)

                # Mostrar en pantalla
                cv2.imshow("RTSP Stream", processed_frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Tecla 'q' presionada. Saliendo...")
                    break

            # FPS final
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                actual_fps = frame_count / elapsed_time
                print(f"FPS promedio de visualización: {actual_fps:.2f}")

    except Exception as e:
        print(f"Error al abrir o procesar el stream de video: {e}")
    finally:
        # Limpiar recursos
        if 'out' in locals() and out is not None and out.isOpened():
            out.release()
        cv2.destroyAllWindows()
        print("Ventanas de OpenCV cerradas.")

    print("Script terminado.")
