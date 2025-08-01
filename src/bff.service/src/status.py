import os
import subprocess
import json
import psutil
import time

other_services = [
    "vhs.detection.service",
    "vhs.sync.db.service"
]

actions = [
    "start",
    "stop",
    "restart"
]

def restart_service(action, service_name):
    try:
        if action not in actions:
            raise ValueError(f"Action {action} is not recognized.")
        # No necesitamos verificar si el servicio está en una lista fija ahora
        subprocess.run(["sudo", "systemctl", action, service_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restart_service: {e}")
        return False

# Función para obtener el estado del sistema
def get_system_status():

    def get_vhs_camera_services():
        output = subprocess.getoutput("systemctl list-units --plain --no-legend --no-pager 'vhs.camera@*.service'")
        services = [line.strip().split()[0] for line in output.strip().split('\n') if line.strip()]
        return services

    def get_service_status(service_name):
        status_output = subprocess.getoutput(f"systemctl status {service_name}")
        status_lines = status_output.splitlines()

        status_info = {
            "service_name": service_name,
            "status": status_lines[0] if status_lines else "Unknown",
            "details": status_lines[1:] if len(status_lines) > 1 else []
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
            temp_output = subprocess.getoutput("vcgencmd measure_temp")
            return temp_output if "temp=" in temp_output else "Not available"
        except Exception as e:
            return str(e)

    def get_fan_status():
        return "ON"  # Modificar si hay un método real para obtener el estado del ventilador

    vhs_camera_services = get_vhs_camera_services()
    all_services_to_check = vhs_camera_services + other_services
    servicesStatus = [get_service_status(service) for service in all_services_to_check]

    status = {
        "memory": get_memory_status(),
        "swap": get_swap_status(),
        "cpu": get_cpu_status(),
        "temperature": get_temperature(),
        "fan_status": get_fan_status(),
        "services": servicesStatus
    }

    return status