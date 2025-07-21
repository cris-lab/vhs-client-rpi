#!/bin/bash
set -e

# 1. Actualizar el sistema
echo "🔄 Actualizando sistema..."
sudo apt update && sudo apt full-upgrade -y

# 2. Actualizar firmware si es necesario
echo "🧬 Verificando versión de firmware..."
FIRMWARE_DATE=$(sudo rpi-eeprom-update | grep -oP '\d{1,2} \w+ \d{4}')
if [[ $(date -d "$FIRMWARE_DATE" +%s) -lt $(date -d "6 December 2023" +%s) ]]; then
  echo "⚠️ Firmware desactualizado. Actualizando..."
  sudo raspi-config nonint do_bootloader_update E1
  sudo rpi-eeprom-update -a
  sudo reboot
fi

# 3. Instalar TightVNC Server y XFCE4
sudo apt install -y tightvncserver xfce4 xfce4-clipman mousepad git hailo-all nginx

# 4. Lanzar clipboard y editor solo si está en entorno gráfico (opcional)
if [ "$DISPLAY" ]; then
  xfce4-clipman &
  mousepad &
fi

# 5. Clonar o actualizar repositorio
echo "📦 Clonando o actualizando VHS..."
if [ ! -d "/opt/vhs/.git" ]; then
  sudo git clone https://github.com/cris-lab/vhs-client-rpi.git /opt/vhs
else
  cd /opt/vhs
  sudo git pull
fi

# 6. Crear carpetas necesarias
echo "📁 Creando carpetas de datos..."
sudo mkdir -p /opt/vhs/storage/detections
sudo mkdir -p /var/lib/vhs/detections

# 7. Permisos
sudo chown -R kakashi:kakashi /opt/vhs
sudo chown -R kakashi:kakashi /var/lib/vhs

# 8. Configurar NGINX permisos
echo "🔧 Configurando permisos NGINX..."
sudo usermod -aG www-data kakashi
sudo chmod -R 775 /var/www/html
sudo chown -R www-data:www-data /var/www/html

# 9. Instalar Tailscale si no existe
echo "🌐 Instalando Tailscale..."
if ! command -v tailscale &>/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

# 10. Crear y usar virtualenv
if [ ! -d "/opt/vhs/env" ]; then
  python3 -m venv /opt/vhs/env
fi

source /opt/vhs/env/bin/activate
pip install --upgrade pip
pip install -r /opt/vhs/requirements.txt
deactivate

# 11. Asegura permisos en xstartup
chmod +x ~/.vnc/xstartup

echo "✅ Instalación finalizada. Reinicia con: sudo reboot"
