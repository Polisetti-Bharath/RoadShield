import os
import logging
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO

from PIL import Image

from sample_utils.report import (
    estimate_severity,
    render_authority_contact_settings,
    render_location_picker,
    render_report_card,
)
from sample_utils.ui import (
    CLASS_COLORS,
    inject_base_css,
    render_footer,
    render_page_header,
)

st.set_page_config(
    page_title="Video Detection - RoadShield",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_base_css()

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "models" / "YOLOv8_Small_RDD.pt"


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


net = load_yolo_model(str(MODEL_LOCAL_PATH))

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


# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False


# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())


def processVideo(video_file, score_threshold):

    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)

    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS) or 25.0
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        _duration = _frame_count/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds).zfill(2)

        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown(
                f'<div class="rs-stat"><div class="rs-stat-value">{_duration_strings}</div>'
                f'<div class="rs-stat-label">Duration (min:sec)</div></div>',
                unsafe_allow_html=True,
            )
        with info_cols[1]:
            st.markdown(
                f'<div class="rs-stat"><div class="rs-stat-value">{_width}x{_height}</div>'
                f'<div class="rs-stat-label">Resolution</div></div>',
                unsafe_allow_html=True,
            )
        with info_cols[2]:
            st.markdown(
                f'<div class="rs-stat"><div class="rs-stat-value">{_fps:.0f}</div>'
                f'<div class="rs-stat-label">FPS</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        detection_totals = Counter()
        best_pothole = None  # (severity_pct, severity_label, annotated_frame_rgb)

        # Read until video is completed
        _frame_counter = 0
        while videoCapture.isOpened():
            ret, frame = videoCapture.read()
            if not ret:
                inferenceBar.empty()
                break

            # frame is native BGR from cv2.VideoCapture
            # YOLO predict expects BGR for numpy arrays, letterboxes internally to 640x640
            results = net.predict(frame, conf=score_threshold, imgsz=640, verbose=False)

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
            detection_totals.update(d.label for d in detections)

            for det in detections:
                if det.label != "Potholes":
                    continue
                severity, severity_pct = estimate_severity(det.box, _width, _height)
                if best_pothole is None or severity_pct > best_pothole[0]:
                    best_pothole = (severity_pct, severity, None)

            # YOLO plot() returns BGR image with drawn bounding boxes at native resolution
            annotated_frame_bgr = results[0].plot()

            # For Streamlit display (expects RGB)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

            if best_pothole is not None and best_pothole[2] is None:
                best_pothole = (best_pothole[0], best_pothole[1], annotated_frame_rgb.copy())

            # Write the BGR image to output video
            cv2writer.write(annotated_frame_bgr)

            # Display the RGB image in Streamlit
            imageLocation.image(annotated_frame_rgb)

            _frame_counter += 1
            if _frame_count > 0:
                inferenceBar.progress(min(1.0, _frame_counter / _frame_count), text=inferenceBarText)

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    st.session_state["video_best_pothole"] = best_pothole

    # Download button for the video
    st.success("Video processed successfully!")

    st.write("")
    st.markdown('<p class="rs-section-label">Damage detected across the video</p>', unsafe_allow_html=True)
    if detection_totals:
        stat_cols = st.columns(len(detection_totals))
        for col, (label, n) in zip(stat_cols, detection_totals.items()):
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
        st.info(f"No damage detected across any frame at confidence threshold {score_threshold:.2f}. Try lowering the confidence threshold.")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        # Also rerun the appplication after download
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="⬇ Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                width="stretch",
            )

    with col2:
        if st.button('Restart', width="stretch", type="primary"):
            # Rerun the application
            st.rerun()

    if best_pothole is not None:
        severity_pct, severity, frame_rgb = best_pothole
        st.write("")
        st.markdown('<p class="rs-section-label">Report to Authorities</p>', unsafe_allow_html=True)
        st.caption("Showing the most severe pothole frame found in this video.")

        location = render_location_picker(key_prefix="vid")
        if location:
            lat, lon, address = location
            whatsapp_number, authority_email = render_authority_contact_settings()
            render_report_card(
                key_prefix="vid_report",
                label="Potholes",
                severity=severity,
                severity_pct=severity_pct,
                lat=lat,
                lon=lon,
                address=address,
                image=Image.fromarray(frame_rgb),
                whatsapp_number=whatsapp_number,
                authority_email=authority_email,
            )
        else:
            st.info("Share your location above to enable reporting.")


render_page_header(
    icon="🎬",
    title="Video Detection",
    subtitle="Upload a drive-through recording and RoadShield will scan it frame by frame for damage.",
)

with st.container(key="rs-upload-card"):
    video_file = st.file_uploader("Upload a video (.mp4)", type=".mp4", disabled=st.session_state.runningInference)
    st.caption("There is a 1GB limit for video size. Resize or trim your video if it's larger than that.")

    score_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        disabled=st.session_state.runningInference,
        help="Default is 0.25. Lower if damage isn't being detected; raise if you see false positives."
    )
    st.caption("Default is 0.25. Lower the threshold if damage isn't being detected (cracks and potholes in drive-through video often have subtle textures).")

if video_file is not None:
    st.write("")
    if st.button('▶ Process Video', width="stretch", disabled=st.session_state.runningInference, type="primary", key="processing_button"):
        st.warning(f"Processing {video_file.name}...")
        processVideo(video_file, score_threshold)

render_footer()
