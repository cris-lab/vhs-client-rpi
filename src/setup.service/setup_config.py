import json
import sys
import os
import uuid
from dotenv import load_dotenv

# Función: Cargar un JSON desde ruta
def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"INFO: No se encontró {config_path}.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: El archivo '{config_path}' no es un JSON válido.")
        sys.exit(1)

# Función: ¿Hay stream configurado?
def is_stream_configured():
    return all([
        os.getenv('STREAM_CODE', '').strip(),
        os.getenv('STREAM_NAME', '').strip(),
        os.getenv('STREAM_URL', '').strip()
    ])

# Cargar variables desde el .env real del sistema
ENV_PATH = '/opt/vhs/src/setup.service/.env'
load_dotenv(ENV_PATH)

# --- Determinar si sobrescribir o no ---
existing_config = load_config('/var/lib/vhs/config.json')

if existing_config:
    print("Ya existe un archivo de configuración en /var/lib/vhs/config.json")
    if os.getenv("REPLACE_CUSTOM_CONFIG", "no").strip().lower() != "si":
        print("REPLACE_CUSTOM_CONFIG no es 'si'. No se reemplazará la configuración existente.")
        sys.exit(0)
    else:
        print("REPLACE_CUSTOM_CONFIG='si'. Sobrescribiendo configuración existente.")
        config = existing_config
else:
    print("No se encontró configuración previa. Creando una nueva.")
    config = load_config('/opt/vhs/src/setup.service/config_template.json')
    if config is None:
        print("ERROR: No se pudo cargar config_template.json. Abortando.")
        sys.exit(1)

# Asegurarse de que streams sea una lista
if 'streams' not in config or not isinstance(config['streams'], list):
    config['streams'] = []

# Insertar datos de tienda
config['code'] = os.getenv("STORE_CODE", "")
config['name'] = os.getenv("STORE_NAME", "")

# Insertar stream (si está configurado)
if is_stream_configured():
    stream_template = load_config('/opt/vhs/src/setup.service/config_template_stream.json')
    if stream_template:
        stream_id = str(uuid.uuid4())
        stream_template['id'] = stream_id
        stream_template['code'] = os.getenv('STREAM_CODE')
        stream_template['name'] = os.getenv('STREAM_NAME')
        stream_template['input']['url'] = os.getenv('STREAM_URL')
        config['streams'].append(stream_template)
        print(f"Stream '{stream_template['name']}' agregado con ID: {stream_id}")

        # Guardar el stream_id para que el bash lo lea después
        with open('/var/lib/vhs/stream_id.txt', 'w') as f:
            f.write(stream_id)

    else:
        print("ERROR: No se pudo cargar config_template_stream.json. Stream no será agregado.")


# Guardar configuración final
try:
    with open('/var/lib/vhs/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("Archivo de configuración guardado exitosamente en /var/lib/vhs/config.json")
except Exception as e:
    print(f"ERROR al guardar el archivo de configuración: {e}")
    sys.exit(1)
