#!/bin/bash

# --- Configuración de Logging ---
LOG_FILE="/var/log/setup_config.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==================================================="
echo "Iniciando script de configuración: $(date)"
echo "==================================================="

# Ruta al archivo .env en el pendrive
CONFIG_FILE="/mnt/config_usb/.env" # O "/mnt/config_usb/.env"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Archivo de configuración $CONFIG_FILE no encontrado. Saliendo."
    exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

echo "Cargando configuración desde $CONFIG_FILE..."

# --- Configuración del Sistema ---

if [ -n "$TIMEZONE" ]; then
    echo "Configurando zona horaria a $TIMEZONE..."
    sudo timedatectl set-timezone "$TIMEZONE"
else
    echo "TIMEZONE no definido en .env. Se mantiene la zona horaria actual."
fi

if [ -n "$HOSTNAME" ] && [ "$(hostname)" != "$HOSTNAME" ]; then
    echo "Configurando hostname a $HOSTNAME..."
    sudo hostnamectl set-hostname "$HOSTNAME"
    sudo sed -i "s/127.0.1.1\s.*/127.0.1.1\t$HOSTNAME/" /etc/hosts
    echo "Hostname cambiado a $HOSTNAME. Puede requerir un reinicio completo para reflejarse en todos los servicios."
else
    echo "HOSTNAME no definido en .env o es igual al actual. Se mantiene el hostname."
fi

# --- Configuración de Red (Optimizado para NetworkManager) ---

echo "Configurando interfaz de red: $NETWORK_INTERFACE con método: $NETWORK_METHOD..."

# Verificar si NetworkManager está activo
if ! systemctl is-active --quiet NetworkManager; then
    echo "Error: NetworkManager no está activo. Este script está optimizado para NetworkManager. Saliendo de la configuración de red."
    # Aquí podrías añadir una lógica de fallback si quisieras, pero para simplificar, se asume NetworkManager.
else
    echo "NetworkManager está activo. Procediendo con la configuración de red."

    # Eliminar cualquier conexión de NetworkManager existente para esta interfaz.
    # Esto evita conflictos con configuraciones previas (ej. de DHCP a estática o viceversa).
    # Primero listamos las conexiones y luego las borramos si pertenecen a la interfaz.
    CONNECTIONS_TO_DELETE=$(nmcli -t -f UUID,DEVICE con show | grep "$NETWORK_INTERFACE" | cut -d: -f1)
    if [ -n "$CONNECTIONS_TO_DELETE" ]; then
        echo "Eliminando conexiones NetworkManager existentes para $NETWORK_INTERFACE: $CONNECTIONS_TO_DELETE"
        for conn_uuid in $CONNECTIONS_TO_DELETE; do
            sudo nmcli con delete "$conn_uuid"
        done
    else
        echo "No se encontraron conexiones NetworkManager existentes para $NETWORK_INTERFACE."
    fi

    # Configuración DHCP
    if [ "$NETWORK_METHOD" = "dhcp" ] || [ -z "$NETWORK_METHOD" ]; then
        echo "Configurando $NETWORK_INTERFACE para DHCP..."
        if sudo nmcli con add type ethernet ifname "$NETWORK_INTERFACE" con-name "$NETWORK_INTERFACE-dhcp" ipv4.method auto; then
            echo "Conexión DHCP para $NETWORK_INTERFACE creada/activada exitosamente."
            sudo nmcli con up "$NETWORK_INTERFACE-dhcp"
        else
            echo "Error al configurar DHCP para $NETWORK_INTERFACE con NetworkManager."
        fi
    # Configuración Estática
    elif [ "$NETWORK_METHOD" = "static" ]; then
        echo "Configurando $NETWORK_INTERFACE con IP estática: $STATIC_IP..."
        if [ -z "$STATIC_IP" ] || [ -z "$STATIC_NETMASK" ] || [ -z "$STATIC_GATEWAY" ] || [ -z "$STATIC_DNS" ]; then
            echo "Error: Faltan parámetros para la configuración IP estática en el archivo .env."
        else
            # Convertir netmask a prefijo CIDR (ej. 255.255.255.0 -> 24)
            IFS='.' read -r i1 i2 i3 i4 <<< "$STATIC_NETMASK"
            NETMASK_BIN=$(printf "%s%s%s%s" \
                $(echo "obase=2; $i1" | bc) \
                $(echo "obase=2; $i2" | bc) \
                $(echo "obase=2; $i3" | bc)
                $(echo "obase=2; $i4" | bc))
            STATIC_CIDR=$(echo "$NETMASK_BIN" | grep -o '1' | wc -l)

            # Crear o modificar la conexión estática
            if sudo nmcli con add type ethernet ifname "$NETWORK_INTERFACE" con-name "$NETWORK_INTERFACE-static" \
                    ipv4.method manual ipv4.addresses "$STATIC_IP/$STATIC_CIDR" \
                    ipv4.gateway "$STATIC_GATEWAY" ipv4.dns "$(echo $STATIC_DNS | sed 's/ /,/g')"; then
                echo "Conexión estática para $NETWORK_INTERFACE creada/activada exitosamente."
                sudo nmcli con up "$NETWORK_INTERFACE-static"
            else
                echo "Error al configurar IP estática para $NETWORK_INTERFACE con NetworkManager."
            fi
        fi
    else
        echo "Advertencia: NETWORK_METHOD '$NETWORK_METHOD' no reconocido. No se configurará la interfaz $NETWORK_INTERFACE."
    fi

    # Configuración de Wi-Fi (si se especifica y NETWORK_INTERFACE es wlan0)
    if [ "$NETWORK_INTERFACE" = "wlan0" ] && [ -n "$SSID" ] && [ -n "$PSK" ]; then
        echo "Configurando Wi-Fi para SSID: $SSID..."
        # nmcli automáticamente manejará la adición/activación de la conexión Wi-Fi.
        if sudo nmcli dev wifi connect "$SSID" password "$PSK" ifname "$NETWORK_INTERFACE"; then
            echo "Conexión Wi-Fi a '$SSID' establecida exitosamente vía NetworkManager."
        else
            echo "Error al conectar Wi-Fi a '$SSID' con NetworkManager. Verifique SSID/PSK o los logs."
        fi
    elif [ "$NETWORK_INTERFACE" = "wlan0" ]; then
        echo "Advertencia: NETWORK_INTERFACE es wlan0, pero SSID o PSK están vacíos en .env. No se configurará Wi-Fi."
    fi

    echo "La configuración de red con NetworkManager se ha intentado. El sistema debería estar conectado."
else
    echo "Saltando la configuración de red avanzada ya que NetworkManager no está activo."
fi


# --- Configuración de Tailscale (Opcional) ---
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    echo "Configurando Tailscale..."
    if ! command -v tailscale &> /dev/null; then
        echo "Tailscale no encontrado, intentando instalar..."
        curl -fsSL https://tailscale.com/install.sh | sh
        if [ $? -ne 0 ]; then
            echo "Error: No se pudo instalar Tailscale. Continuará sin él."
        fi
    fi

    if command -v tailscale &> /dev/null; then
        TAILSCALE_COMMAND="sudo tailscale up --authkey=$TAILSCALE_AUTH_KEY"

        if [ -n "$TAILSCALE_HOSTNAME" ]; then
            TAILSCALE_COMMAND="$TAILSCALE_COMMAND --hostname=$TAILSCALE_HOSTNAME"
        fi

        if [ -n "$TAILSCALE_TAGS" ]; then
            TAILSCALE_COMMAND="$TAILSCALE_COMMAND --advertise-tags=$(echo $TAILSCALE_TAGS | sed 's/,/,/g')"
        fi

        if [ "$TAILSCALE_NO_EPHEMERAL" = "true" ]; then
            TAILSCALE_COMMAND="$TAILSCALE_COMMAND --reusable-key"
        fi

        echo "Ejecutando comando Tailscale: $TAILSCALE_COMMAND"
        $TAILSCALE_COMMAND

        if [ $? -eq 0 ]; then
            echo "Tailscale configurado exitosamente."
        else
            echo "Error al configurar Tailscale. Revisa la clave de autenticación, las etiquetas o los logs."
        fi
    else
        echo "Tailscale no está instalado o no se pudo ejecutar, no se realizará la configuración de Tailscale."
    fi
else
    echo "TAILSCALE_AUTH_KEY no definido en .env. No se configurará Tailscale."
fi

echo "==================================================="
echo "Script de configuración finalizado: $(date)"
echo "==================================================="

exit 0