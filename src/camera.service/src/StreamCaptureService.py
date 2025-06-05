import numpy as np
import subprocess
import threading
import asyncio
import cv2, os, json, yt_dlp

class StreamCaptureService:

    def __init__(self, stream_url=None, fps=None, dimensions=None):
        
        print("Inicializando StreamCaptureService")
        print(f"Stream URL: {stream_url}")
        print(f"FPS: {fps}")
        print(f"Dimensiones: {dimensions}")
        
        self.stream_url = stream_url
        self.stop_event = threading.Event()
        self.fps = int(fps)
        self.dimensions = tuple(dimensions)

    async def start_stream(self, callback=None):
        """Inicia la captura del stream"""
        print("Iniciando captura de stream")

        # Si es YouTube, obtener el stream_url real
        if 'youtube.com' in self.stream_url or 'youtu.be' in self.stream_url:
            print("Detectado enlace de YouTube, extrayendo URL de stream...")
            self.stream_url = await self.extract_youtube_stream(self.stream_url)
        else:
            self.stream_url = self.stream_url

        if not self.stream_url:
            print("No se pudo obtener la URL de stream. Abortando.")
            return
        
        await self.capture_with_ffmpeg(self.stream_url, callback)

    async def extract_youtube_stream(self, url):
        """Usa yt-dlp para extraer la mejor URL de stream"""
        ydl_opts = {
            'quiet': True,
            'format': 'best[ext=mp4]/best',
            'noplaylist': True,
        }

        loop = asyncio.get_event_loop()
        stream_url = None

        def get_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('url')

        try:
            stream_url = await loop.run_in_executor(None, get_info)
            print(f"URL extraída: {stream_url}")
        except Exception as e:
            print(f"Error al extraer URL de YouTube: {e}")

        return stream_url

    async def capture_with_ffmpeg(self, stream_url, callback=None):
        width, height = self.dimensions
        buffer_size = width * height * 3  # Resolución WxH, 3 bytes por píxel (RGB24)

        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-rtbufsize", "1M",
            "-i", stream_url,
            "-vf", f"fps={self.fps},scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-an",
            "-rtsp_transport", "udp",
            "-threads", "1",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-max_delay", "0",
            "-analyzeduration", "0",
            "-probesize", "8192",
            "-strict", "experimental",
            "pipe:1"
        ]
        
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error al iniciar FFmpeg: {e}")
            return
        
        empty_frame_count = 0

        while not self.stop_event.is_set():
            try:
                raw_frame = process.stdout.read(buffer_size)
                
                if len(raw_frame) == 0:
                    empty_frame_count += 1
                    if empty_frame_count > 10:
                        print("Demasiados frames vacíos. Terminando...")
                        break
                    continue

                empty_frame_count = 0  # Reiniciar si recibimos frames válidos

                # Crear un array a partir del buffer de FFmpeg
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                if frame.size != buffer_size:
                    print("Frame recibido con tamaño incorrecto. Descartando...")
                    continue

                frame = frame.reshape((height, width, 3))
                if callback:
                    await callback(frame)

            except Exception as e:
                print(f"Error al procesar frame con FFmpeg: {e}")
                break
        
        process.terminate()
        process.wait()
        print("Proceso FFmpeg terminado.")

    def is_frame_valid(self, frame, threshold=30):
        if frame is None or not isinstance(frame, np.ndarray):
            return False

        # Convertir a escala de grises y calcular la desviación estándar
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_frame)

        # Si la desviación estándar es muy baja, la imagen es posiblemente gris uniforme
        if std_dev < threshold:
            print(f"Desviación estándar: {std_dev}")
            print("Frame descartado por baja variabilidad (imagen gris o estática).")
            return False
        
        print(f"Desviación estándar: {std_dev}")
        return True

    def stop(self):
        """Detiene la captura del stream"""
        self.stop_event.set()
        if self.rtsp_process:
            self.rtsp_process.terminate()
            self.rtsp_process.wait()
        print("Captura detenida.")
