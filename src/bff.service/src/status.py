import os
import subprocess
import json
import psutil
import time

# Función para obtener el estado del sistema
def get_system_status():
    
    def get_service_status(service_name):
        status_output = subprocess.getoutput(f"systemctl status {service_name}")
        status_lines = status_output.splitlines()
        
        status_info = {
            "service_name": service_name,
            "status": status_lines[0],
            "details": status_lines[1:]
        }
        return status_info

    def get_memory_status():
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }

    def get_swap_status():
        swap = psutil.swap_memory()
        return {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent
        }

    def get_cpu_status():
        return {
            "percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count(logical=False)
        }

    def get_temperature():
        try:
            temp = subprocess.getoutput("vcgencmd measure_temp")
            return temp
        except Exception as e:
            return str(e)

    def get_fan_status():
        try:
            fan_status = "ON"
            return fan_status
        except Exception as e:
            return str(e)

    status = {
        "memory": get_memory_status(),
        "swap": get_swap_status(),
        "cpu": get_cpu_status(),
        "temperature": get_temperature(),
        "fan_status": get_fan_status(),
        "services": [
            get_service_status("vhs.age.estimation.service"),
            get_service_status("vhs.gender.classification.service"),
            get_service_status("vhs.head.detection.service"),
            get_service_status("vhs.face.analyze.service"),
            get_service_status("vhs.sync.db.service")
        ]
    }
    
    return status