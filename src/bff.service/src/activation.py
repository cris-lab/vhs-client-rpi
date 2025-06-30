import json
import subprocess
import uuid
from settings import get_config_from_json, CONFIG_PATH

def get_device_id():
    
    config = get_config_from_json()
    if "id" in config:
        if config["id"] is not None:
            return config["id"]
    raise ValueError("Device ID not found in configuration.")

#import os
def activate_device(activation_code):
    config = get_config_from_json()

    if not activation_code:
        raise ValueError("Activation code is required.")

    if config.get("id"):
        raise ValueError("Device is already activated.")

    # Simulación de validación de código
    if activation_code == "1234":
        device_id = str(uuid.uuid4())
        config.update({
            "id": device_id,
            "code": "",
            "name": "",
            "time_zone": "UTC",
            "detection_schedule": {
                "enabled": False,
                "start_time": "08:00",
                "end_time": "21:00"
            },
            "streams": []
        })

        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)

        return device_id  # Puedes retornar el ID si es útil
    else:
        raise ValueError("Invalid activation code provided.")

    