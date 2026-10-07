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


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(query: str):
    """Converts a street name, city, or landmark into (lat, lon, display_name)."""
    if not query or not query.strip():
        return None
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query.strip(), "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "RoadShield-PBL-Project"},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            first = results[0]
            return float(first["lat"]), float(first["lon"]), first.get("display_name")
    except Exception:
        pass
    return None


def get_network_location():
    """Auto-detects device/network location via IP geolocation when browser GPS is blocked."""
    try:
        r = requests.get("http://ip-api.com/json", timeout=3)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                city = d.get("city", "")
                region = d.get("regionName", "")
                country = d.get("country", "")
                addr = ", ".join(filter(None, [city, region, country]))
                return float(d.get("lat")), float(d.get("lon")), addr
    except Exception:
        pass

    try:
        r = requests.get("https://ipapi.co/json/", timeout=3)
        if r.status_code == 200:
            d = r.json()
            city = d.get("city", "")
            region = d.get("region", "")
            country = d.get("country_name", "")
            addr = ", ".join(filter(None, [city, region, country]))
            return float(d.get("latitude")), float(d.get("longitude")), addr
    except Exception:
        pass

    return 17.3850, 78.4867, "Hyderabad, Telangana, India"


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


def render_location_picker(key_prefix: str, default_address: str = "Hyderabad, Telangana, India"):
    """Renders a comprehensive location-capture widget with auto-detection,
    browser GPS, address search, and manual coordinates."""
    state_key = f"{key_prefix}_location"

    # Auto-initialize location if not already set
    if state_key not in st.session_state or st.session_state[state_key] is None:
        net_loc = get_network_location()
        if net_loc:
            st.session_state[state_key] = net_loc
        else:
            st.session_state[state_key] = (17.3850, 78.4867, default_address)

    current_val = st.session_state[state_key]
    if len(current_val) == 3:
        lat, lon, custom_addr = current_val
    else:
        lat, lon = current_val
        custom_addr = None

    display_addr = custom_addr or (reverse_geocode(lat, lon) if (lat and lon) else default_address)

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 8px;">
            <span style="color: #10B981; font-weight: 600;">📍 Active Location:</span>
            <span style="color: #F8FAFC;"> {display_addr}</span>
            <span style="color: #94A3B8; font-size: 0.85em; margin-left: 8px;">({lat:.4f}, {lon:.4f})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Change Location / Auto-Detect Settings", expanded=False):
        tab1, tab2, tab3 = st.tabs(["🎯 Auto-Detect", "🔍 Search Address / Road", "✏️ Manual Coordinates"])

        with tab1:
            st.caption("Click below to re-detect your live location via Network / IP or Browser GPS:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🎯 Auto-Detect My Location Now", key=f"{key_prefix}_autodetect_btn", width="stretch", type="primary"):
                    with st.spinner("Detecting your location..."):
                        detected = get_network_location()
                        if detected:
                            st.session_state[state_key] = detected
                            st.success(f"Location detected: {detected[2]}")
                            st.rerun()
                        else:
                            st.error("Could not automatically determine location. Please use Search or Coordinates tab.")
            with c2:
                st.caption("Browser Hardware GPS:")
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") is not None:
                    browser_lat = float(loc["latitude"])
                    browser_lon = float(loc["longitude"])
                    if abs(browser_lat - lat) > 0.0001 or abs(browser_lon - lon) > 0.0001:
                        addr = reverse_geocode(browser_lat, browser_lon)
                        st.session_state[state_key] = (browser_lat, browser_lon, addr)
                        st.rerun()

        with tab2:
            st.caption("Type any road name, area, or landmark (e.g., 'Banjara Hills', 'MG Road', 'Connaught Place'):")
            search_col, btn_col = st.columns([3, 1])
            with search_col:
                search_query = st.text_input("Enter location to search", placeholder="e.g. Hitec City, Hyderabad", key=f"{key_prefix}_search_input", label_visibility="collapsed")
            with btn_col:
                if st.button("🔍 Search", key=f"{key_prefix}_search_btn", width="stretch"):
                    if search_query:
                        with st.spinner("Searching location..."):
                            geo_res = geocode_address(search_query)
                            if geo_res:
                                st.session_state[state_key] = geo_res
                                st.success(f"Found: {geo_res[2]}")
                                st.rerun()
                            else:
                                st.warning("Location not found. Please try a different query.")

        with tab3:
            st.caption("Enter precise latitude and longitude:")
            m_lat = st.number_input("Latitude", value=float(lat), format="%.6f", key=f"{key_prefix}_lat_num")
            m_lon = st.number_input("Longitude", value=float(lon), format="%.6f", key=f"{key_prefix}_lon_num")
            m_addr = st.text_input("Custom Address Label (Optional)", value=display_addr or "", key=f"{key_prefix}_addr_custom")
            if st.button("💾 Apply Coordinates", key=f"{key_prefix}_apply_coords_btn", width="stretch"):
                st.session_state[state_key] = (m_lat, m_lon, m_addr if m_addr else None)
                st.rerun()

    return lat, lon, display_addr


def build_complaint_pdf(
    label: str,
    severity: str,
    severity_pct: float,
    lat,
    lon,
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

    coords_str = f"{lat:.6f}, {lon:.6f}" if (lat is not None and lon is not None) else "Not provided"
    maps_link = (
        f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        if (lat is not None and lon is not None)
        else "N/A"
    )

    rows = [
        ("Damage type", label),
        ("Severity", f"{severity} (~{severity_pct:.1f}% of frame)"),
        ("Date / time", timestamp.strftime("%d %b %Y, %I:%M %p")),
        ("Coordinates", coords_str),
        ("Address", address or "Not available"),
        ("Map link", maps_link),
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
    lat=None,
    lon=None,
    address=None,
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
    if lat is not None and lon is not None:
        maps_link = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        location_text = address or f"{lat:.6f}, {lon:.6f}"
    else:
        maps_link = "https://maps.google.com"
        location_text = address or "Location not specified"

    message = (
        f"Road damage report ({label}, {severity} severity) detected via RoadShield.\n"
        f"Location: {location_text}\n"
        f"Map: {maps_link}\n"
        f"Reported: {timestamp.strftime('%d %b %Y, %I:%M %p')}\n"
        f"Please look into repairing this at the earliest."
    )

    st.markdown(f"📍 **{location_text}**")
    if lat is not None and lon is not None:
        st.caption(f"Severity: {severity} (~{severity_pct:.1f}% of frame) · [Open in Google Maps]({maps_link})")
    else:
        st.caption(f"Severity: {severity} (~{severity_pct:.1f}% of frame)")

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
