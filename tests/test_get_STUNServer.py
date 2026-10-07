"""Unit tests for sample_utils/get_STUNServer.py's failure fallback.

The success path makes three live HTTP calls (geoip cache, IP list, user
geolocation) and isn't exercised here -- only that a failure anywhere in
that chain falls back cleanly instead of crashing the Realtime Detection page.
"""

from unittest.mock import patch

from sample_utils.get_STUNServer import FALLBACK_STUN_SERVER, getSTUNServer


def test_getSTUNServer_falls_back_on_request_failure():
    with patch("sample_utils.get_STUNServer.requests.get", side_effect=ConnectionError("network down")):
        assert getSTUNServer() == FALLBACK_STUN_SERVER
