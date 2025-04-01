from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

async def get_events(
    model_name: str = 'default',
    from_date: int = 0,
    to_date: int = 0
    
):
    