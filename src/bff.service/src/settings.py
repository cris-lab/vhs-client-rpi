import json
import subprocess

CONFIG_PATH = "/var/lib/vhs/config.json"

def get_config_from_json():
    try:
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Archivo de configuración no encontrado. Creando uno vacío en {CONFIG_PATH}.")
        try:
            with open(CONFIG_PATH, "w") as file:
                json.dump({}, file, indent=2)
        except Exception as write_err:
            raise Exception(f"No se pudo crear el archivo de configuración vacío: {write_err}")
        return {}
    except json.JSONDecodeError:
        raise Exception(f"Error decoding JSON from the configuration file {CONFIG_PATH}.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def update_settings(payload):
    
    config = get_config_from_json()
    
    print("Updating settings with payload:",payload)
    
    if "code" in payload:
        
        config["code"] = payload["code"]
        
    if "name" in payload:
        config["name"] = payload["name"]
        
    if "time_zone" in payload:
        config["time_zone"] = payload["time_zone"]
    
    if "detection_schedule" in payload:
        if "enabled" in payload["detection_schedule"]:
            config["detection_schedule"]["enabled"] = payload["detection_schedule"]["enabled"]
            
        if "start_time" in payload["detection_schedule"]:
            config["detection_schedule"]["start_time"] = payload["detection_schedule"]["start_time"]
            
        if "end_time" in payload["detection_schedule"]:
            config["detection_schedule"]["end_time"] = payload["detection_schedule"]["end_time"]
         
    # Guardar los cambios en config.json si hubo modificaciones
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)

    return config
