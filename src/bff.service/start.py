from fastapi import FastAPI, Request, Response, WebSocket, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from sse_starlette.sse import EventSourceResponse
from collections import deque
import uvicorn, logging, asyncio, json, cv2, io, os
from src.status import get_system_status, restart_service
from src.config import get_config_from_json, update_config, check_cnn_url
from src.settings import update_settings
from src.settings_streams import update_stream_settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
active_websockets = {}
app = FastAPI()

# Habilitar CORS para permitir solicitudes desde localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes desde localhost:4200
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

FILE_PATH = '/var/lib/vhs'
DETECTIONS_BASE_DIR = "/opt/vhs/storage/detections"

# @app.get("/thumbnail/{stream_id}")
# async def get_thumbnail(stream_id: str):
#     thumbnail_path = _get_thumbnail_path(stream_id, FILE_PATH)

#     if not os.path.exists(thumbnail_path):
#         raise HTTPException(status_code=404, detail="Thumbnail no encontrado.")

#     return StreamingResponse(open(thumbnail_path, "rb"), media_type="image/jpeg")

@app.get("/models")
async def get_models():
    models_dir = "/opt/vhs/src/api.models/models"
    # ls models_dir
    try:
        models = os.listdir(models_dir)
        return {"models": models}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Models directory not found")

@app.post("/restart")
async def restart_service_endpoint(request: Request, response:Response):
    response.status_code = 400
    payload = await request.json()
    service_name = payload.get("service")
    action = payload.get("action")
    if service_name and action:
        success = restart_service(action, service_name)
        if success:
            response.status_code = 200
            return {"message": f"Service {service_name}:{action} successfully."}
        
    return {"error": "Service name not provided."}
    
@app.get("/status")
def get_status():
    return get_system_status()  # No necesitas JSONResponse()

@app.get("/settings")
def get_config():
    return get_config_from_json()

@app.put("/settings/{stream_id}")
async def set_config(request: Request, stream_id: str):
    payload = await request.json()
    return update_stream_settings(stream_id, payload)

@app.post("/processed_stream/{stream_id}")
async def receive_processed_frame(stream_id: str, request: Request):
    websocket = active_websockets.get(stream_id)
    if websocket and websocket.client_state == WebSocketState.CONNECTED:
        try:
            image_bytes = await request.body()
            await websocket.send_bytes(image_bytes)
            return {"status": "frame sent"}
        except Exception as e:
            print(f"Error sending to websocket for {stream_id}: {e}")
            return {"status": "websocket send error"}
    return {"status": "no active websocket for this stream"}

@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    active_websockets[stream_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # o mantener el socket abierto
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        del active_websockets[stream_id]
     
BUFFER_SIZE = 1
events_buffer = {} # Almacenamos los buffers por stream_id     
        
@app.post('/processed_events/{stream_id}')
async def receive_processed_events(stream_id: str, request: Request):
    try:
        event_data = await request.json()
        
        # Verificar si ya existe un buffer para este stream_id, si no, crear uno nuevo
        if stream_id not in events_buffer:
            events_buffer[stream_id] = deque(maxlen=BUFFER_SIZE)

        # Agregar el nuevo evento al buffer, el deque elimina automáticamente el más antiguo
        events_buffer[stream_id].append(event_data)

        return {"status": "event stored", "stream_id": stream_id, "event_data": event_data}

    except Exception as e:
        print(f"Error receiving event for {stream_id}: {e}")
        return {"status": "error", "message": str(e)}



@app.get("/events/{stream_id}")
async def get_events(stream_id: str):
    if stream_id not in events_buffer:
        events_buffer[stream_id] = deque(maxlen=BUFFER_SIZE)

    async def event_stream():
        try:
            while True:
                if events_buffer[stream_id]:
                    event = events_buffer[stream_id].popleft()
                    yield f"{json.dumps(event)}\n\n" 
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print(f"🔌 Cliente SSE desconectado: {stream_id}")
            return

    return EventSourceResponse(
        event_stream(),
        headers={"Access-Control-Allow-Origin": "*"}
    )
  
  
  
@app.get("/detections")
async def list_detections(request: Request): # Añadimos 'request' para obtener la URL base
    """
    Lista todas las detecciones de personas guardadas en la estructura UUID/data.json.
    Retorna un array de objetos JSON, donde cada objeto contiene los datos de una persona detectada.
    Incluye URLs para las imágenes de los ROIs.
    """
    detections = []
    if not os.path.exists(DETECTIONS_BASE_DIR):
        logger.warning(f"Directorio de detecciones no encontrado: {DETECTIONS_BASE_DIR}")
        return {"detections": []}

    try:
        # Obtener la URL base de la aplicación para construir los enlaces de las imágenes
        # Esto considera si estás usando un proxy inverso o un puerto específico
        base_url = str(request.base_url).rstrip('/')

        # Recorrer cada directorio que representa un UUID de persona
        for person_uuid_dir in os.listdir(DETECTIONS_BASE_DIR):
            person_dir_path = os.path.join(DETECTIONS_BASE_DIR, person_uuid_dir)

            # Asegurarse de que sea un directorio válido (un UUID)
            if os.path.isdir(person_dir_path):
                data_json_path = os.path.join(person_dir_path, 'data.json')
                images_dir_path = os.path.join(person_dir_path, 'images')

                if os.path.exists(data_json_path):
                    try:
                        with open(data_json_path, 'r') as f:
                            person_data = json.load(f)

                            # Añadir un nodo 'image_urls' con los enlaces a las imágenes
                            image_urls = []
                            if os.path.exists(images_dir_path) and os.path.isdir(images_dir_path):
                                for image_filename in os.listdir(images_dir_path):
                                    # Filtrar solo archivos de imagen si es necesario (ej. .jpg, .png)
                                    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                        # Construir la URL completa para cada imagen
                                        # Ej: http://localhost:8000/detections/UUID/images/roi_timestamp.jpg
                                        image_url = f"{base_url}/detections/{person_uuid_dir}/images/{image_filename}"
                                        image_urls.append(image_url)
                            
                            # Opcional: Podrías ordenar las imágenes por timestamp si el nombre del archivo lo permite
                            # image_urls.sort() 

                            person_data['image_urls'] = image_urls
                            detections.append(person_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decodificando JSON en {data_json_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error leyendo o procesando {data_json_path}: {e}")
                else:
                    logger.warning(f"No se encontró data.json en el directorio: {person_dir_path}")

    except Exception as e:
        logger.error(f"Error al listar detecciones en {DETECTIONS_BASE_DIR}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el directorio de detecciones: {e}")

    return {"detections": detections}


@app.get("/detections/{person_uuid}/images/{image_filename}")
async def get_detection_image(person_uuid: str, image_filename: str):
    """
    Sirve una imagen de detección específica (ROI) dado el UUID de la persona
    y el nombre del archivo de la imagen.
    """
    image_path = os.path.join(DETECTIONS_BASE_DIR, person_uuid, 'images', image_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Imagen de detección no encontrada.")

    return StreamingResponse(open(image_path, "rb"), media_type="image/jpeg")