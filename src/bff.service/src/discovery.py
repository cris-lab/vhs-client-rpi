#!/usr/bin/env python3
from onvif import WsDiscoveryClient

def discover_onvif():
    """
    Descubre cámaras ONVIF en la red usando onvif-client.
    """
    wsd = WsDiscoveryClient()
    devices = wsd.search()   # ← sin pasar timeout aquí
    if not devices:
        print("No se encontraron cámaras ONVIF.")
        return

    print(f"Encontradas {len(devices)} cámaras ONVIF:")
    for dev in devices:
        print("— XAddrs:", dev.xaddrs)
        print("  Types: ", dev.types)
        print("  Scopes:", dev.scopes)
        print()
    wsd.dispose()

if __name__ == "__main__":
    discover_onvif()
