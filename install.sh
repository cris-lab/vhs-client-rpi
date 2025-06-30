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
sudo mkdir -p /var/lib/vhs/{detections,events}
sudo chown -R kakashi:kakashi /var/lib/vhs

# 8. Activar systemd services desde cada módulo
echo "⚙️ Configurando servicios de systemd..."
find /opt/vhs/src -type f -path "*/src/systemd/*" | while read service_file; do
  service_name=$(basename "$service_file")
  sudo cp "$service_file" "/etc/systemd/system/$service_name"
  sudo systemctl daemon-reexec
  sudo systemctl enable "$service_name"
done

# 9. Instalar Tailscale
echo "🌐 Instalando Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh

# 10. Asignar hostname único basado en UUID
echo "🔤 Asignando hostname único..."
UUID=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)
HOSTNAME="vhs-rpi-$UUID"
echo "$HOSTNAME" | sudo tee /etc/hostname
sudo hostnamectl set-hostname "$HOSTNAME"
sudo sed -i "s/127.0.1.1.*/127.0.1.1\t$HOSTNAME/" /etc/hosts

echo "✅ Instalación finalizada. Reinicia con: sudo reboot"
