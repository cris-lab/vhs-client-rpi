import json
import subprocess

CONFIG_PATH = "/var/lib/vhs/config.json"
FRAME_PATH = "/var/lib/vhs/frame.jpg"
TEST_FRAME_PATH = "/var/lib/vhs/test_frame.jpg"

SIZES = {
    0: [640,480],
    1: [1280,720],
    2: [1920,1080],
    3: [2040,1080]
}

MIN_FPS = 1
MAX_FPS = 10

TRACKER_MIN_THRESHOLD = 0.1
TRACKER_MAX_THRESHOLD = 3
TRACKER_MIN_LOST_FRAMES = 1
TRACKER_MAX_LOST_FRAMES = 20

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

    config = get_config_from_json()  # Obtener la configuración actual

    # Si 'url' está presente, la actualiza y guarda el primer frame
    if "url" in payload["input"]:   
        url = input_data["url"]
        cmd = ["ffmpeg", "-i", url, "-frames:v", "1", "-q:v", "2", FRAME_PATH]
        try:
            subprocess.run(cmd, check=True)
            payload["input"]
        except subprocess.CalledProcessError:
            print("Error al capturar el frame con ffmpeg")
    # Si 'size' está presente, la actualiza
    if "size" in payload["input"]:
        if input_data["size"] in [0,1,2,3]:
            config["input"]["size"] = SIZES[input_data["size"]]
    # Si 'fps' está presente, la actualiza
    if "fps" in payload["input"]:
        if MIN_FPS <= input_data["fps"] <= MAX_FPS:
            config["input"]["fps"] = input_data["fps"]
            
    # Si 'tracker' está presente, la actualiza
    if "min_threshold" in input_data["tracker"]:
        if TRACKER_MIN_THRESHOLD <= input_data["tracker"]["min_threshold"] <= TRACKER_MAX_THRESHOLD:
            config["input"]["tracker"] = config.get("input", {}).get("tracker", {})
            config["input"]["tracker"]["min_threshold"] = input_data["tracker"]["min_threshold"]
    if "max_threshold" in input_data["tracker"]:
        if TRACKER_MIN_THRESHOLD <= input_data["tracker"]["max_threshold"] <= TRACKER_MAX_THRESHOLD:
            config["input"]["tracker"] = config.get("input", {}).get("tracker", {})
            config["input"]["tracker"]["max_threshold"] = input_data["tracker"]["max_threshold"]
    if "max_lost_frames" in input_data["tracker"]:
        if TRACKER_MIN_LOST_FRAMES <= input_data["tracker"]["max_lost_frames"] <= TRACKER_MAX_LOST_FRAMES:
            config["input"]["tracker"] = config.get("input", {}).get("tracker", {})
            config["input"]["tracker"]["max_lost_frames"] = input_data["tracker"]["max_lost_frames"]
    
    if payload.get("in_out_interpolation") in [0,1]:
        config["input"]["in_out_interpolation"] = payload["in_out_interpolation"]      
    

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