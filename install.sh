#!/bin/bash
set -e

# 1. Actualizar el sistema
echo "🔄 Actualizando sistema..."
sudo apt update && sudo apt full-upgrade -y

# 2. Actualizar firmware si es necesario
echo "🧬 Verificando versión de firmware..."
# Extraer la línea con la fecha
FIRMWARE_LINE=$(sudo rpi-eeprom-update | grep "CURRENT:")
echo "FIRMWARE_LINE: $FIRMWARE_LINE"

# Extraer la fecha completa
FIRMWARE_DATE=$(echo "$FIRMWARE_LINE" | cut -d':' -f2 | cut -d'(' -f1 | xargs)
echo "FIRMWARE_DATE: $FIRMWARE_DATE"

# Validar si la fecha es menor al mínimo requerido
MIN_DATE="6 December 2023"

if [[ $(date -d "$FIRMWARE_DATE" +%s) -lt $(date -d "$MIN_DATE" +%s) ]]; then
  echo "⚠️ Firmware desactualizado. Actualizando..."
  sudo raspi-config nonint do_bootloader_update E1
  sudo rpi-eeprom-update -a
  sudo reboot
else
  echo "✅ Firmware actualizado: $FIRMWARE_DATE"
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
  sudo git config --global --add safe.directory /opt/vhs
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

if ! command -v tailscale &>/dev/null; then
  echo "🌐 Instalando Tailscale..."
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

echo "✅ Instalación finalizada. Reinicia con: sudo reboot"

echo "fin
