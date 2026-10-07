import logging
from collections import Counter
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from sample_utils.logging_config import configure_logging
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

configure_logging()
inject_base_css()

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "models" / "YOLOv8_Small_RDD.pt"

net = load_model(MODEL_LOCAL_PATH)

render_page_header(
    icon="🖼️",
    title="Image Detection",
    subtitle="Upload a photo of a road surface and RoadShield will highlight every crack or pothole it finds.",
)

with st.container(key="rs-upload-card"):
    image_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])
    score_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Default is 0.25. Lower if damage is faint or not detected; raise if you see false positives.",
    )
    st.caption(
        "Default is 0.25. Lower the threshold if damage isn't being detected "
        "(cracks and potholes often have lower confidence scores)."
    )

    st.markdown("**Or test with a sample image:**")
    sample_cols = st.columns(4)
    if "selected_sample" not in st.session_state:
        st.session_state["selected_sample"] = None

    samples_dir = ROOT / "resource" / "samples"
    with sample_cols[0]:
        if st.button("Alligator Crack", width="stretch"):
            st.session_state["selected_sample"] = samples_dir / "alligator_crack.jpg"
    with sample_cols[1]:
        if st.button("Longitudinal Crack", width="stretch"):
            st.session_state["selected_sample"] = samples_dir / "longitudinal_crack.jpg"
    with sample_cols[2]:
        if st.button("Transverse Crack", width="stretch"):
            st.session_state["selected_sample"] = samples_dir / "transverse_crack.jpg"
    with sample_cols[3]:
        if st.button("Pothole & Crack", width="stretch"):
            st.session_state["selected_sample"] = samples_dir / "pothole_and_crack.jpg"

image_source = None
image_bytes = None
source_name = "sample"

if image_file is not None:
    image_source = image_file
    image_bytes = image_file.getvalue()
    source_name = image_file.name
    st.session_state["selected_sample"] = None
elif st.session_state.get("selected_sample") and Path(st.session_state["selected_sample"]).exists():
    sample_path = Path(st.session_state["selected_sample"])
    image_source = sample_path
    image_bytes = sample_path.read_bytes()
    source_name = sample_path.name
    st.info(f"Loaded sample image: **{sample_path.name}**")

if image_source is not None:
    try:
        raw_image = Image.open(image_source)
        raw_image.load()
        # Apply EXIF rotation (if any from mobile cameras) and ensure 3-channel RGB
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
    except Exception:
        logger.exception("Failed to decode image from source %s", source_name)
        st.error("Couldn't read this file as an image. Please upload a valid PNG or JPG.")
        st.stop()

    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]

    with st.spinner("Running detection..."):
        try:
            # Predict using PIL Image directly so YOLO applies letterboxing and preserves aspect ratio
            results = net.predict(image, conf=score_threshold, imgsz=640, verbose=False)
        except Exception:
            logger.exception("Image inference failed for %s", source_name)
            st.error("Detection failed on this image. Please try a different file.")
            st.stop()

    # Save the results
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for _box in boxes:
            cls_id = int(_box.cls[0]) if hasattr(_box.cls, "__len__") else int(_box.cls)
            conf_val = float(_box.conf[0]) if hasattr(_box.conf, "__len__") else float(_box.conf)
            xyxy_val = _box.xyxy[0].astype(int) if len(_box.xyxy.shape) > 1 else _box.xyxy.astype(int)
            label_name = net.names.get(cls_id, CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}")
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=label_name,
                    score=conf_val,
                    box=xyxy_val,
                )
            )

    logger.info("Image %s: %d detection(s)", source_name, len(detections))

    # YOLO plot() returns a BGR numpy array on the original image canvas
    annotated_frame_bgr = results[0].plot()
    _image_pred = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

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
        st.info(
            f"No damage detected at confidence threshold {score_threshold:.2f}. "
            "Try lowering the confidence threshold slider above (e.g. to 0.15–0.20) if the damage is faint."
        )

    st.write("")
    col1, col2 = st.columns(2)

    # Original Image
    with col1:
        st.markdown("**Original**")
        st.image(_image, width="stretch")

    # Predicted Image
    with col2:
        st.markdown(f"**Detected damage ({len(detections)} bounding boxes)**")
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

    if detections:
        st.write("")
        st.markdown('<p class="rs-section-label">Report to Authorities</p>', unsafe_allow_html=True)

        exif_location = get_gps_from_exif(image_bytes)
        if exif_location and "img_location" not in st.session_state:
            exif_lat, exif_lon = exif_location
            exif_addr = reverse_geocode(exif_lat, exif_lon)
            st.session_state["img_location"] = (exif_lat, exif_lon, exif_addr)
            st.caption("📍 Coordinates automatically read from photo's EXIF GPS metadata.")

        lat, lon, address = render_location_picker(key_prefix="img")

        whatsapp_number, authority_email = render_authority_contact_settings()

        sorted_detections = sorted(
            detections,
            key=lambda d: estimate_severity(d.box, w_ori, h_ori)[1],
            reverse=True,
        )

        for i, det in enumerate(sorted_detections):
            severity, severity_pct = estimate_severity(det.box, w_ori, h_ori)
            icon = "🚨" if det.label == "Potholes" else "⚠️"
            with st.expander(
                f"{icon} {det.label} #{i + 1} — {severity} severity (~{severity_pct:.1f}% area)",
                expanded=(i == 0),
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

render_footer()
