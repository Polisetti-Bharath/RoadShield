import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Deep learning framework
from ultralytics import YOLO

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

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


render_page_header(
    icon="🖼️",
    title="Image Detection",
    subtitle="Upload a photo of a road surface and RoadShield will highlight every crack or pothole it finds.",
)

with st.container():
    st.markdown('<div class="rs-card">', unsafe_allow_html=True)
    image_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg"])
    score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.caption("Lower the threshold if damage isn't being detected. Raise it if you're seeing false positives.")
    st.markdown('</div>', unsafe_allow_html=True)

if image_file is not None:

    # Load the image
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

render_footer()
