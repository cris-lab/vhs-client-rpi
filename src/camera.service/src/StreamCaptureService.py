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

        if not self.stream_url:
            print("No se pudo obtener la URL de stream. Abortando.")
            return

        # Intentos de reconexión si falla
        max_retries = 5
        for attempt in range(max_retries):
            print(f"Conectando intento #{attempt + 1}...")
            try:
                await self.capture_with_ffmpeg(self.stream_url, callback)
                break  # Éxito
            except Exception as e:
                print(f"Error durante la captura: {e}")
                await asyncio.sleep(5)
        else:
            print("Demasiados intentos fallidos. Abortando.")

    async def extract_youtube_stream(self, url):
        """Usa yt-dlp para extraer la mejor URL de stream"""
        ydl_opts = {
            'quiet': True,
            'format': 'best[ext=mp4]/best',
            'noplaylist': True,
        }

        loop = asyncio.get_event_loop()

        def get_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('url')

        try:
            stream_url = await loop.run_in_executor(None, get_info)
            print(f"URL extraída: {stream_url}")
            return stream_url
        except Exception as e:
            print(f"Error al extraer URL de YouTube: {e}")
            return None

    async def capture_with_ffmpeg(self, stream_url, callback=None):
        width, height = self.dimensions
        buffer_size = width * height * 3  # RGB24

        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", stream_url,
            "-vf", f"fps={self.fps},scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-sn",
            "-"
        ]

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Captura de stderr en paralelo para evitar bloqueo
        async def read_stderr():
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                print(f"[FFmpeg stderr] {line.decode(errors='ignore').strip()}")

        asyncio.create_task(read_stderr())

        try:
            while not self.stop_event.is_set():
                raw_frame = await process.stdout.readexactly(buffer_size)

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

                if callback:
                    await callback(frame)

        except asyncio.IncompleteReadError:
            print("⚠️ FFmpeg terminó la lectura de frames (posible desconexión del stream).")
            # Intentar leer lo último de stderr para diagnóstico
            try:
                remaining = await process.stderr.read()
                if remaining:
                    print("⚠️ FFmpeg stderr final:\n", remaining.decode(errors='ignore'))
            except Exception as e:
                print("No se pudo leer stderr final:", e)

            raise  # Forzar manejo en el bloque de reconexión

        except Exception as e:
            print(f"❌ Error inesperado en captura: {e}")
            raise

        finally:
            process.terminate()
            await process.wait()
            print("✅ Proceso FFmpeg terminado.")

    def is_frame_valid(self, frame, threshold=30):
        if frame is None or not isinstance(frame, np.ndarray):
            return False

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_frame)

        if std_dev < threshold:
            print(f"Desviación estándar: {std_dev}")
            print("⚠️ Frame descartado por baja variabilidad (imagen gris o estática).")
            return False

        print(f"Desviación estándar: {std_dev}")
        return True

    def stop(self):
        """Detiene la captura del stream"""
        self.stop_event.set()
        print("🛑 Captura detenida.")
