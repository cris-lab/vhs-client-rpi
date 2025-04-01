from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from src.status import get_system_status
from src.config import get_config_from_json, update_config, check_cnn_url

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Permite solicitudes desde localhost:4200
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

@app.get("/thumbnail")
def get_thumbnail():
    frame = "/var/lib/vhs/frame.jpg"
    headers = {"Content-Type": "image/jpeg"}
    try:
        with open(frame, "rb") as image_file:
            image_data = image_file.read()
        return Response(content=image_data, media_type="image/jpeg", headers=headers)
    except FileNotFoundError:
        logger.error(f"File {frame} not found.")
        return {"error": "File not found."}

@app.get("/status")
def get_status():
    return get_system_status()  # No necesitas JSONResponse()

@app.get("/settings")
def get_config():
    return get_config_from_json()

@app.put("/settings")
async def set_config(request: Request):
    payload = await request.json()
    return update_config(payload)

@app.post("/settings/check_cnn_url")
async def check(request: Request):
    payload = await request.json()
    return check_cnn_url(payload)
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
