"""Helpers for turning a pothole detection into a complaint that can be sent
to Indian municipal authorities (WhatsApp, email, or a downloadable PDF for
manual filing on portals like CPGRAMS)."""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import piexif
import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image
from streamlit_geolocation import streamlit_geolocation

CONTACT_FILE = Path(__file__).resolve().parent.parent / ".rs_authority_contact.json"


def load_saved_contact():
    """Loads the previously saved authority WhatsApp/email, if any."""
    try:
        data = json.loads(CONTACT_FILE.read_text())
        return data.get("whatsapp", ""), data.get("email", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return "", ""


def save_contact(whatsapp: str, email: str) -> None:
    """Persists the authority WhatsApp/email to disk so it's remembered next time."""
    CONTACT_FILE.write_text(json.dumps({"whatsapp": whatsapp, "email": email}))


def render_authority_contact_settings():
    """Renders a one-time 'default authority contact' widget and returns
    (whatsapp_number, email) to reuse across every report card on the page."""
    if "authority_whatsapp" not in st.session_state:
        saved_whatsapp, saved_email = load_saved_contact()
        st.session_state["authority_whatsapp"] = saved_whatsapp
        st.session_state["authority_email"] = saved_email

    with st.expander("⚙️ Authority contact (used for all reports below)", expanded=False):
        whatsapp = st.text_input(
            "WhatsApp number (with country code, e.g. 91XXXXXXXXXX)",
            value=st.session_state["authority_whatsapp"],
            key="authority_whatsapp_input",
        )
        email = st.text_input(
            "Email address",
            value=st.session_state["authority_email"],
            key="authority_email_input",
        )
        if st.button("💾 Remember this contact"):
            st.session_state["authority_whatsapp"] = whatsapp
            st.session_state["authority_email"] = email
            save_contact(whatsapp, email)
            st.success("Saved. This will be pre-filled next time you open RoadShield.")

    return whatsapp, email


def get_gps_from_exif(image_bytes: bytes):
    """Returns (lat, lon) from a photo's EXIF GPS tags, or None if unavailable."""
    try:
        exif_dict = piexif.load(image_bytes)
    except Exception:
        return None

    gps = exif_dict.get("GPS")
    if not gps:
        return None

    def _to_degrees(dms, ref):
        try:
            d = dms[0][0] / dms[0][1]
            m = dms[1][0] / dms[1][1]
            s = dms[2][0] / dms[2][1]
        except (IndexError, ZeroDivisionError):
            return None
        deg = d + m / 60 + s / 3600
        if ref in (b"S", b"W"):
            deg = -deg
        return deg

    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E")

    if not (lat_dms and lon_dms):
        return None

    lat = _to_degrees(lat_dms, lat_ref)
    lon = _to_degrees(lon_dms, lon_ref)
    if lat is None or lon is None:
        return None
    return lat, lon


@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat: float, lon: float):
    """Human-readable address for a coordinate, via OpenStreetMap Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2"},
            headers={"User-Agent": "RoadShield-PBL-Project"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("display_name")
    except Exception:
        return None


def estimate_severity(box, frame_width: int, frame_height: int):
    """Rough severity from how much of the frame the damage box covers."""
    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_width * frame_height
    pct = (box_area / frame_area) * 100 if frame_area else 0.0
    if pct >= 6:
        return "High", pct
    if pct >= 2:
        return "Medium", pct
    return "Low", pct


def render_location_picker(key_prefix: str):
    """Renders a location-capture widget. Returns (lat, lon, address) once resolved,
    or None while the user still needs to grant location access / enter it manually."""
    state_key = f"{key_prefix}_location"
    st.session_state.setdefault(state_key, None)

    if st.session_state[state_key] is None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("Tap to share your current location:")
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") is not None:
                st.session_state[state_key] = (loc["latitude"], loc["longitude"])
                st.rerun()
        with col2:
            with st.popover("Or enter coordinates manually"):
                lat = st.number_input(
                    "Latitude", value=0.0, format="%.6f", key=f"{key_prefix}_lat_manual"
                )
                lon = st.number_input(
                    "Longitude", value=0.0, format="%.6f", key=f"{key_prefix}_lon_manual"
                )
                if st.button("Use these coordinates", key=f"{key_prefix}_manual_btn"):
                    st.session_state[state_key] = (lat, lon)
                    st.rerun()
        return None

    lat, lon = st.session_state[state_key]
    address = reverse_geocode(lat, lon)
    return lat, lon, address


def build_complaint_pdf(
    label: str,
    severity: str,
    severity_pct: float,
    lat: float,
    lon: float,
    address,
    timestamp: datetime,
    image: Image.Image,
) -> bytes:
    """Builds a one-page PDF complaint report, ready to attach on CPGRAMS or a
    local municipal grievance portal."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Road Damage Complaint Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Generated by RoadShield", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    img_buffer = BytesIO()
    image.convert("RGB").save(img_buffer, format="JPEG")
    img_buffer.seek(0)
    pdf.image(img_buffer, w=170)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Details", ln=True)
    pdf.set_font("Helvetica", "", 11)

    rows = [
        ("Damage type", label),
        ("Severity", f"{severity} (~{severity_pct:.1f}% of frame)"),
        ("Date / time", timestamp.strftime("%d %b %Y, %I:%M %p")),
        ("Coordinates", f"{lat:.6f}, {lon:.6f}"),
        ("Address", address or "Not available"),
        ("Map link", f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"),
    ]
    for key, value in rows:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(40, 7, f"{key}:")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, str(value))

    return bytes(pdf.output())


def render_report_card(
    *,
    key_prefix: str,
    label: str,
    severity: str,
    severity_pct: float,
    lat: float,
    lon: float,
    address,
    image: Image.Image,
    whatsapp_number: str = "",
    authority_email: str = "",
) -> None:
    """Renders the 'Report to Authorities' card: WhatsApp / email / PDF options.

    There's no single public API for filing a road-damage complaint in India —
    every municipal body runs its own portal/app — so this hands the user a
    ready-made complaint through the channels that actually work: WhatsApp
    grievance lines (common with corporations like BBMP/MCGM), email, or a PDF
    to attach when filing on CPGRAMS (pgportal.gov.in) or a local portal.

    whatsapp_number/authority_email come from render_authority_contact_settings(),
    set once per page and reused for every report card on it.
    """
    timestamp = datetime.now()
    maps_link = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

    message = (
        f"Road damage report ({label}, {severity} severity) detected via RoadShield.\n"
        f"Location: {address or f'{lat:.6f}, {lon:.6f}'}\n"
        f"Map: {maps_link}\n"
        f"Reported: {timestamp.strftime('%d %b %Y, %I:%M %p')}\n"
        f"Please look into repairing this at the earliest."
    )

    st.markdown(f"📍 **{address or f'{lat:.6f}, {lon:.6f}'}**")
    st.caption(f"Severity: {severity} (~{severity_pct:.1f}% of frame) · [Open in Google Maps]({maps_link})")

    if not whatsapp_number and not authority_email:
        st.caption("No authority contact set yet — open '⚙️ Authority contact' above to add one.")

    col1, col2, col3 = st.columns(3)
    with col1:
        wa_url = f"https://wa.me/{whatsapp_number}?text={quote(message)}" if whatsapp_number else f"https://wa.me/?text={quote(message)}"
        st.link_button("💬 Send via WhatsApp", wa_url, width="stretch")
    with col2:
        mailto_url = (
            f"mailto:{authority_email}?subject={quote('Road Damage Complaint - ' + label)}"
            f"&body={quote(message)}"
        )
        st.link_button("✉ Send via Email", mailto_url, width="stretch")
    with col3:
        pdf_bytes = build_complaint_pdf(
            label, severity, severity_pct, lat, lon, address, timestamp, image
        )
        st.download_button(
            "⬇ Download PDF",
            data=pdf_bytes,
            file_name=f"RoadShield_Complaint_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"{key_prefix}_pdf",
        )

    st.caption(
        "No WhatsApp/email on hand? File the downloaded PDF on the national grievance "
        "portal: [pgportal.gov.in](https://pgportal.gov.in) (CPGRAMS)."
    )
