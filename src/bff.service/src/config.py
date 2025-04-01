import json
import subprocess

CONFIG_PATH = "/var/lib/vhs/config.json"
FRAME_PATH = "/var/lib/vhs/frame.jpg"
TEST_FRAME_PATH = "/var/lib/vhs/test_frame.jpg"

def get_config_from_json():
    try:
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Devuelve un diccionario vacío en lugar de lanzar un error
    except json.JSONDecodeError:
        raise Exception(f"Error decoding JSON from the configuration file {CONFIG_PATH}.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def update_config(payload):
    
    if "input" not in payload:
        return  # Si no hay 'input', no hacemos nada

    input_data = payload["input"]
    config = get_config_from_json()  # Obtener la configuración actual

    # Si 'url' está presente, la actualiza y guarda el primer frame
    if "url" in input_data:
        url = input_data["url"]
        
        cmd = ["ffmpeg", "-i", url, "-frames:v", "1", "-q:v", "2", FRAME_PATH]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("Error al capturar el frame con ffmpeg")

        config.setdefault("input", {})  # Asegurar que 'input' existe
        config["input"]["url"] = url  # Actualizar solo 'url'

    # Si 'size' está presente, la actualiza
    if "size" in input_data:
        config.setdefault("input", {}) 
        config["input"]["size"] = input_data["size"]

    # Si 'fps' está presente, la actualiza
    if "fps" in input_data:
        config.setdefault("input", {}) 
        config["input"]["fps"] = input_data["fps"]

    # Guardar los cambios en config.json si hubo modificaciones
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)

    return config

def check_cnn_url(payload):
    
    print(payload)
    
    if "url" in payload:
        url = payload["url"]
        cmd = ["ffmpeg", "-i", url, "-frames:v", "1", "-q:v", "2", TEST_FRAME_PATH]
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
        
    return False