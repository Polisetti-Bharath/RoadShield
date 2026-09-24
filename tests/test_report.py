"""Unit tests for the pure-logic helpers in sample_utils/report.py.

UI-rendering functions (render_*) and network calls (reverse_geocode) are
intentionally not covered here -- they need a running Streamlit script
context / live network access to exercise meaningfully.
"""

from datetime import datetime
from io import BytesIO

import numpy as np
import piexif
import pytest
from PIL import Image

from sample_utils.report import build_complaint_pdf, estimate_severity, get_gps_from_exif


class TestEstimateSeverity:
    def test_small_box_is_low_severity(self):
        # 10x10 box in a 640x640 frame => ~0.02% of the frame
        severity, pct = estimate_severity((0, 0, 10, 10), 640, 640)
        assert severity == "Low"
        assert pct < 2

    def test_medium_box_is_medium_severity(self):
        # ~3% of a 640x640 frame
        box = (0, 0, 140, 140)
        severity, pct = estimate_severity(box, 640, 640)
        assert severity == "Medium"
        assert 2 <= pct < 6

    def test_large_box_is_high_severity(self):
        # ~10% of a 640x640 frame
        box = (0, 0, 260, 260)
        severity, pct = estimate_severity(box, 640, 640)
        assert severity == "High"
        assert pct >= 6

    def test_zero_area_frame_does_not_divide_by_zero(self):
        severity, pct = estimate_severity((0, 0, 10, 10), 0, 0)
        assert severity == "Low"
        assert pct == 0.0

    def test_degenerate_box_coordinates_clamped_to_zero_area(self):
        # x2 < x1 / y2 < y1 shouldn't produce a negative area
        severity, pct = estimate_severity((50, 50, 10, 10), 640, 640)
        assert severity == "Low"
        assert pct == 0.0


class TestGetGpsFromExif:
    def _build_jpeg_with_gps(self, lat: float, lon: float) -> bytes:
        def _to_dms(value):
            deg = int(value)
            minutes_full = (value - deg) * 60
            minutes = int(minutes_full)
            seconds = round((minutes_full - minutes) * 60 * 100)
            return [(deg, 1), (minutes, 1), (seconds, 100)]

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _to_dms(abs(lat)),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _to_dms(abs(lon)),
        }
        exif_bytes = piexif.dump({"GPS": gps_ifd})

        buffer = BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="JPEG", exif=exif_bytes)
        return buffer.getvalue()

    def test_extracts_lat_lon_from_exif(self):
        image_bytes = self._build_jpeg_with_gps(12.9716, 77.5946)
        result = get_gps_from_exif(image_bytes)

        assert result is not None
        lat, lon = result
        assert lat == pytest.approx(12.9716, abs=1e-2)
        assert lon == pytest.approx(77.5946, abs=1e-2)

    def test_returns_none_when_no_gps_data(self):
        buffer = BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
        assert get_gps_from_exif(buffer.getvalue()) is None

    def test_returns_none_on_invalid_bytes(self):
        assert get_gps_from_exif(b"not a real image") is None


class TestBuildComplaintPdf:
    def test_returns_non_empty_pdf_bytes(self):
        image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        pdf_bytes = build_complaint_pdf(
            label="Potholes",
            severity="High",
            severity_pct=8.3,
            lat=12.9716,
            lon=77.5946,
            address="MG Road, Bengaluru",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            image=image,
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")
