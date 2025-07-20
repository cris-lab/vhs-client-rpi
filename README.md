# Servicio de Sincronización

## Configuración de Eventos a Base de Datos

1. Dentro de `src/sync.service`, crea un archivo `.env` con las siguientes variables de entorno:

   ```ini
   MONGO_URI=
   MONGO_DB_NAME=
   ```
---

## Configuración de VPN con Tailscale

1. Inicia Tailscale con:

   ```bash
   sudo tailscale up
   ```

   Esto responderá con un mensaje similar a:

   ```
   To authenticate, visit:
   https://login.tailscale.com/a/sdjfweofiweof
   ```

2. Verifica si el servicio `tailscaled` está habilitado:

   ```bash
   sudo systemctl is-enabled tailscaled
   ```

   Si el comando devuelve `enabled`, el servicio está correctamente habilitado.  
   En caso contrario, habilítalo con:

   ```bash
   sudo systemctl enable tailscaled
   ```
