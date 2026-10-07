import logging

import requests

logger = logging.getLogger(__name__)

GEO_LOC_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/geoip_cache.txt"
IPV4_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/valid_ipv4s.txt"
GEO_USER_URL = "https://geolocation-db.com/json"

# Used whenever the nearest-server lookup above fails (network issue, one of
# the three services being down, etc.) so Realtime Detection can still start
# a WebRTC session instead of crashing the page.
FALLBACK_STUN_SERVER = "stun.l.google.com:19302"


def getSTUNServer():
    try:
        geoLocs = requests.get(GEO_LOC_URL, timeout=5).json()

        user_data = requests.get(GEO_USER_URL, timeout=5).json()
        latitude, longitude = user_data["latitude"], user_data["longitude"]

        ip_addresses = requests.get(IPV4_URL, timeout=5).text.strip().split('\n')

        def calculate_distance(addr):
            stunLat, stunLon = geoLocs.get(addr.split(':')[0], (0, 0))
            dist = ((latitude - stunLat) ** 2 + (longitude - stunLon) ** 2) ** 0.5
            return addr, dist

        closest_addr, _ = min(map(calculate_distance, ip_addresses), key=lambda x: x[1])
        logger.info("Resolved nearest STUN server: %s", closest_addr)
        return closest_addr
    except Exception:
        logger.exception(
            "STUN server lookup failed, falling back to %s", FALLBACK_STUN_SERVER
        )
        return FALLBACK_STUN_SERVER
