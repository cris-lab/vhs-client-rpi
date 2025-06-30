import json
import subprocess
from .settings import get_config_from_json, CONFIG_PATH

def update_stream_settings(stream_id, payload):
    config = get_config_from_json()
    streams = config.setdefault('streams', [])

    stream = next((s for s in streams if s.get('id') == stream_id), None)
    if stream is None:
        raise ValueError(f"No se encontró la transmisión con ID {stream_id}")

    # Asegurar existencia de estructuras clave
    stream.setdefault("input", {})
    stream.setdefault("tracker", {})
    stream.setdefault("centroid_orientation", {})

    if "input" in payload:
        if "url" in payload["input"]:
            url = payload["input"]["url"]
            cmd = ["ffmpeg", "-i", url, "-frames:v", "1", "-q:v", "2", "/tmp/frame.jpg"]  # Reemplaza si tienes FRAME_PATH
            try:
                subprocess.run(cmd, check=True)
                stream["input"]["url"] = url
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Error al procesar la URL {url}: {str(e)}")

    if "tracker" in payload:
        t = payload["tracker"].get("track_thresh")
        if t is not None and 0.1 <= t <= 0.9:
            stream["tracker"]["track_thresh"] = t

        m = payload["tracker"].get("match_thresh")
        if m is not None and 0.1 <= m <= 1.0:
            stream["tracker"]["match_thresh"] = m

        b = payload["tracker"].get("track_buffer")
        if b is not None and 1 <= b <= 150:
            stream["tracker"]["track_buffer"] = b

    if "in_out_interpolation" in payload and payload["in_out_interpolation"] in [0, 1]:
        stream["input"]["in_out_interpolation"] = payload["in_out_interpolation"]

    if "centroid_orientation" in payload:
        orientation = payload["centroid_orientation"]
        if "horizontal" in orientation and orientation["horizontal"] in ["left", "center", "right"]:
            stream["centroid_orientation"]["horizontal"] = orientation["horizontal"]
        if "vertical" in orientation and orientation["vertical"] in ["top", "middle", "bottom"]:
            stream["centroid_orientation"]["vertical"] = orientation["vertical"]

    # Guardar cambios
    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)

    return config
