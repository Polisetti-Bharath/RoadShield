import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO

from PIL import Image

from sample_utils.get_STUNServer import getSTUNServer
from sample_utils.report import (
    estimate_severity,
    render_authority_contact_settings,
    render_location_picker,
    render_report_card,
)
from sample_utils.ui import (
    inject_base_css,
    render_footer,
    render_page_header,
)

st.set_page_config(
    page_title="Realtime Detection - RoadShield",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

inject_base_css()

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "models" / "YOLOv8_Small_RDD.pt"

# STUN Server
try:
    STUN_STRING = "stun:" + str(getSTUNServer())
    STUN_SERVER = [{"urls": [STUN_STRING]}]
except Exception:
    STUN_SERVER = [{"urls": ["stun:stun.l.google.com:19302"]}]


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


render_page_header(
    icon="📷",
    title="Realtime Detection",
    subtitle="Detect road damage live using a USB webcam — useful for on-site monitoring with personnel on the ground.",
)

score_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Default is 0.25. Lower if damage isn't being detected; raise if you see false positives."
)
st.caption("Default is 0.25. Cracks and potholes in live camera feeds often have lower confidence scores. Lower the slider if damage is not detected.")

# Holds recent detections for UI display
result_queue: "queue.Queue[List[Detection]]" = queue.Queue(maxsize=10)
# Holds (detections, annotated_frame_rgb, frame_width, frame_height) for reporting
pothole_snapshot_queue: "queue.Queue" = queue.Queue(maxsize=1)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]

    # Run inference directly on native frame preserving aspect ratio
    results = net.predict(image, conf=score_threshold, imgsz=640, verbose=False)

    # Extract detections
    detections: List[Detection] = []
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

    try:
        if result_queue.full():
            result_queue.get_nowait()
        result_queue.put_nowait(detections)
    except Exception:
        pass

    annotated_frame = results[0].plot()

    if detections:
        if pothole_snapshot_queue.full():
            try:
                pothole_snapshot_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            pothole_snapshot_queue.put_nowait(
                (detections, cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), w_ori, h_ori)
            )
        except queue.Full:
            pass

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


st.write("")
webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 640},
        },
        "audio": False
    },
    async_processing=True,
)

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        latest = []
        while not result_queue.empty():
            try:
                latest = result_queue.get_nowait()
            except queue.Empty:
                break
        if latest:
            st.table(latest)
        else:
            st.caption("No damage detected in the latest frame.")
    else:
        st.caption("Start the webcam above to see live predictions here.")

st.divider()
st.markdown('<p class="rs-section-label">Report to Authorities</p>', unsafe_allow_html=True)

if st.button("📸 Capture Road Damage & Report", width="stretch", disabled=not webrtc_ctx.state.playing):
    try:
        st.session_state["rt_pothole_snapshot"] = pothole_snapshot_queue.get_nowait()
    except queue.Empty:
        st.session_state["rt_pothole_snapshot"] = None
        st.warning("No road damage detected yet — keep the camera pointed at the road surface and try again.")

snapshot = st.session_state.get("rt_pothole_snapshot")
if snapshot:
    dets, frame_rgb, f_w, f_h = snapshot
    worst = max(dets, key=lambda d: estimate_severity(d.box, f_w, f_h)[1])
    severity, severity_pct = estimate_severity(worst.box, f_w, f_h)

    location = render_location_picker(key_prefix="rt")
    lat, lon, address = location if location else (12.971599, 77.594566, "Bengaluru, Karnataka, India")

    whatsapp_number, authority_email = render_authority_contact_settings()
    render_report_card(
        key_prefix="rt_report",
        label=worst.label,
        severity=severity,
        severity_pct=severity_pct,
        lat=lat,
        lon=lon,
        address=address,
        image=Image.fromarray(frame_rgb),
        whatsapp_number=whatsapp_number,
        authority_email=authority_email,
    )

render_footer()
