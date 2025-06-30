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
async def set_config(request: Request):
    payload = await request.json()
    return update_stream_settings(stream_id, payload)

@app.post("/settings/check_cnn_url")
async def check(request: Request):
    payload = await request.json()
    return check_cnn_url(payload)

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
     
# Definir el tamaño del buffer (por ejemplo, 100 eventos)
BUFFER_SIZE = 100
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

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
