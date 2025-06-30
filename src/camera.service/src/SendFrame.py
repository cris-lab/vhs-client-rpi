# send_frame.py
import aiohttp

async def SendFrame(session, image_bytes, stream_id, backend_url_base="http://127.0.0.1:8000"):
    backend_url = f"{backend_url_base}/processed_stream/{stream_id}"
    try:
        async with session.post(backend_url, data=image_bytes) as resp:
            await resp.read()
            print(f"Frame enviado a {backend_url} - Status: {resp.status}")
    except aiohttp.ClientError as e:
        print(f"Error enviando frame para {stream_id}: {e}")
    except Exception as e:
        print(f"Error general enviando frame para {stream_id}: {e}")
