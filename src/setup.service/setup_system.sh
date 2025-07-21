#!/bin/bash
# --- Configuración de Logging ---
LOG_FILE="/var/log/setup_config.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==================================================="
echo "Iniciando script de configuración: $(date)"
echo "==================================================="

# Variables principales
PENDRIVE_ENV="/mnt/usb/boot.env"
DEST_ENV="/opt/vhs/src/setup.service/.env"

# --- Validación de archivo .env del pendrive ---
if [ ! -f "$PENDRIVE_ENV" ]; then
    echo "ERROR: No se encontró el archivo de configuración en el pendrive: $PENDRIVE_ENV"
    exit 1
fi

echo "Archivo de configuración encontrado en el pendrive."

# --- Manejo de configuración previa ---
if [ -f "$DEST_ENV" ]; then
    echo "Detectado archivo existente en $DEST_ENV"
    source "$DEST_ENV"
    if [ "$REPLACE_CUSTOM_CONFIG" != "si" ]; then
        echo "REPLACE_CUSTOM_CONFIG distinto de 'si'. Abortando configuración para evitar sobrescritura."
        exit 0
    else
        echo "REPLACE_CUSTOM_CONFIG=si. Sobrescribiendo configuración existente."
    fi
else
    echo "No existe configuración previa. Continuando con configuración inicial."
fi

# --- Copiar archivo .env desde el pendrive ---
cp "$PENDRIVE_ENV" "$DEST_ENV"
if [ $? -eq 0 ]; then
    echo "Archivo .env copiado exitosamente a $DEST_ENV"
else
    echo "ERROR: No se pudo copiar el archivo .env desde $PENDRIVE_ENV a $DEST_ENV"
    exit 1
fi

# --- Cargar configuración actualizada ---
set -a
source "$DEST_ENV"
set +a

echo "Cargando configuración desde $DEST_ENV..."

# --- Configuración del Sistema ---
if [ -n "$TIMEZONE" ]; then
    sudo timedatectl set-timezone "$TIMEZONE"
fi

if [ -n "$HOSTNAME" ] && [ "$(hostname)" != "$HOSTNAME" ]; then
    sudo hostnamectl set-hostname "$HOSTNAME"
    sudo sed -i "s/^127.0.1.1\s.*/127.0.1.1\t$HOSTNAME/" /etc/hosts
fi

# --- Configuración Tailscale ---
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    if ! command -v tailscale &> /dev/null; then
        curl -fsSL https://tailscale.com/install.sh | sh
    fi

    if command -v tailscale &> /dev/null; then
        sudo systemctl enable --now tailscaled
        sleep 5
        TAILSCALE_COMMAND="sudo tailscale up --authkey=$TAILSCALE_AUTH_KEY"
        if [ -n "$TAILSCALE_HOSTNAME" ]; then
            TAILSCALE_COMMAND="$TAILSCALE_COMMAND --hostname=$TAILSCALE_HOSTNAME"
        fi
        if [ -n "$TAILSCALE_TAGS" ]; then
            TAILSCALE_COMMAND="$TAILSCALE_COMMAND --advertise-tags=$(echo "$TAILSCALE_TAGS" | sed 's/ /,/g')"
        fi
        $TAILSCALE_COMMAND
    fi
else
    echo "TAILSCALE_AUTH_KEY no definido. Saltando Tailscale."
fi

# --- Configuración VHS ---
if [ -d "/opt/vhs/env" ]; then
    source /opt/vhs/env/bin/activate
    python3 /opt/vhs/src/setup.service/setup_config.py
fi

if [ -f "/var/lib/vhs/stream_id.txt" ]; then
    STREAM_ID=$(cat /var/lib/vhs/stream_id.txt)
    sudo systemctl enable vhs.camera@"$STREAM_ID".service
fi

echo "==================================================="
echo "Script COMPLETADO: $(date)"
echo "==================================================="

exit 0
