import argparse
import os
import pafy
import time
import cv2
import src.utils as vhs_utils
from src.FrameProcessor import FrameProcessor
from degirum_tools.video_support import (
    open_video_stream,
    video_source,
)

parser              = argparse.ArgumentParser(description='VHS Camera Stream Service')
parser.add_argument('--stream-id', type=str, required=False, help='ID del stream a procesar')

args                = parser.parse_args()
stream_id           = args.stream_id

config              = vhs_utils.load_config(os.path.join('/var/lib/vhs', 'config.json'))

if stream_id is None:
    stream_config       = next((s for s in config.get('streams', []) if s['id'] == stream_id), None)
else:
    stream_config       = config.get('streams', [])[0]  # Default to the first stream if no ID is provided

if stream_config is None:
    raise ValueError(f"No se encontró la transmisión con ID {stream_id}")

video_source_url    = stream_config['input']['url']
frame_processor     = FrameProcessor(config, stream_config, stream_config.get('tracker', {}).get('class_list', ['person', 'head']))

# --- Main Script ---
if __name__ == "__main__":
    print(f"Intentando abrir el stream RTSP: {video_source_url}")

    try:
        
        if "youtube.com" in video_source_url or "youtu.be" in video_source_url:
            video = pafy.new(video_source_url)
            video_source_url = video.getbest(preftype="mp4").url
            print(f"URL resuelta para streaming directo: {video_source_url}")


        with open_video_stream(video_source_url) as stream:
            
            print("Stream de video abierto exitosamente.")

            width   = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height  = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps     = stream.get(cv2.CAP_PROP_FPS)
            print(f"Dimensiones del stream: {width}x{height}, FPS: {fps:.2f}")

            frame_count = 0
            start_time = time.time()
   
            for frame in video_source(stream, fps=30.00):

                roi_frame = vhs_utils.crop_and_resize_roi_padded(frame, stream_config.get('roi', []), target_size=(640, 640))
                processed_frame, _ = frame_processor.execute(roi_frame)
                
                cv2.imshow("RTSP Stream", processed_frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Tecla 'q' presionada. Saliendo...")
                    break

            # Calculate and print average FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > 0:
                actual_fps = frame_count / elapsed_time
                print(f"FPS promedio de visualización: {actual_fps:.2f}")
                

    except Exception as e:
        print(f"Error al abrir o procesar el stream de video: {e}")
    finally:
        # This block will always execute, ensuring resources are cleaned up
        cv2.destroyAllWindows()
        print("Ventanas de OpenCV cerradas.")

    print("Script terminado.")