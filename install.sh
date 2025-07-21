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

# 3. Configurar PCIe Gen 3
echo "⚙️ Configurando PCIe Gen3..."
sudo sed -i '/^dtparam=pciex1_gen=/d' /boot/firmware/config.txt
echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt

# 4. Clonar repositorio
echo "📦 Clonando VHS..."
cd /opt
sudo git clone https://github.com/cris-lab/vhs-client-rpi.git vhs

mkdir -P /opt/vhs/storage/detections

sudo chown -R kakashi:kakashi /opt/vhs

# 5. Instalar dependencias
echo "📥 Instalando paquetes necesarios..."
sudo apt install -y hailo-all nginx

# 6. Configurar permisos en NGINX
echo "🔧 Configurando permisos NGINX..."
sudo usermod -aG www-data kakashi
sudo chmod -R 775 /var/www/html
sudo chown -R www-data:www-data /var/www/html

# 7. Crear carpetas necesarias
echo "📁 Creando carpetas de datos..."
sudo mkdir -p /var/lib/vhs/detections
sudo chown -R kakashi:kakashi /var/lib/vhs


# 9. Instalar Tailscale
echo "🌐 Instalando Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh


python3 -m venv /opt/vhs/env
source /opt/vhs/env/bin/activate
pip install --upgrade pip
pip install -r /opt/vhs/requirements.txt

deactivate

echo "✅ Instalación finalizada. Reinicia con: sudo reboot"