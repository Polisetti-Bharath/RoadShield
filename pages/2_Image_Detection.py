import logging
from collections import Counter
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from sample_utils.model import CLASSES, Detection, load_model
from sample_utils.report import (
    estimate_severity,
    get_gps_from_exif,
    render_authority_contact_settings,
    render_location_picker,
    render_report_card,
    reverse_geocode,
)
from sample_utils.ui import (
    CLASS_COLORS,
    inject_base_css,
    render_footer,
    render_page_header,
)

st.set_page_config(
    page_title="Image Detection - RoadShield",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_base_css()

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"

net = load_model(MODEL_LOCAL_PATH)

render_page_header(
    icon="🖼️",
    title="Image Detection",
    subtitle="Upload a photo of a road surface and RoadShield will highlight every crack or pothole it finds.",
)

with st.container(key="rs-upload-card"):
    image_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg"])
    score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.caption("Lower the threshold if damage isn't being detected. Raise it if you're seeing false positives.")

if image_file is not None:

    # Load the image
    image_bytes = image_file.getvalue()
    image = Image.open(image_file)

    # Perform inference
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    with st.spinner("Running detection..."):
        results = net.predict(image_resized, conf=score_threshold)

    # Save the results
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    st.write("")
    st.markdown('<p class="rs-section-label">Results</p>', unsafe_allow_html=True)

    counts = Counter(d.label for d in detections)
    if counts:
        stat_cols = st.columns(len(counts))
        for col, (label, n) in zip(stat_cols, counts.items()):
            with col:
                color = CLASS_COLORS.get(label, "#2563EB")
                st.markdown(
                    f"""
                    <div class="rs-stat">
                        <div class="rs-stat-value" style="color:{color}">{n}</div>
                        <div class="rs-stat-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("No damage detected in this image at the current confidence threshold.")

    st.write("")
    col1, col2 = st.columns(2)

    # Original Image
    with col1:
        st.markdown("**Original**")
        st.image(_image, width="stretch")

    # Predicted Image
    with col2:
        st.markdown("**Detected damage**")
        st.image(_image_pred, width="stretch")

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="⬇ Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png",
            width="stretch",
        )

    pothole_detections = [d for d in detections if d.label == "Potholes"]
    if pothole_detections:
        st.write("")
        st.markdown('<p class="rs-section-label">Report to Authorities</p>', unsafe_allow_html=True)

        exif_location = get_gps_from_exif(image_bytes)
        if exif_location:
            lat, lon = exif_location
            address = reverse_geocode(lat, lon)
            st.caption("📍 Location auto-detected from the photo's EXIF data.")
        else:
            location = render_location_picker(key_prefix="img")
            lat, lon, address = location if location else (None, None, None)

        if lat is not None:
            whatsapp_number, authority_email = render_authority_contact_settings()
            for i, det in enumerate(pothole_detections):
                severity, severity_pct = estimate_severity(det.box, 640, 640)
                with st.expander(
                    f"🚨 Pothole #{i + 1} — {severity} severity", expanded=(len(pothole_detections) == 1)
                ):
                    render_report_card(
                        key_prefix=f"img_{i}",
                        label=det.label,
                        severity=severity,
                        severity_pct=severity_pct,
                        lat=lat,
                        lon=lon,
                        address=address,
                        image=Image.fromarray(_image_pred),
                        whatsapp_number=whatsapp_number,
                        authority_email=authority_email,
                    )
        else:
            st.info("Share your location above to enable reporting.")

render_footer()
