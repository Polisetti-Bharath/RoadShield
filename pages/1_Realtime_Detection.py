import logging
import queue
from pathlib import Path
from typing import List

import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.get_STUNServer import getSTUNServer
from sample_utils.model import CLASSES, Detection, load_model
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

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"

# STUN Server
STUN_STRING = "stun:" + str(getSTUNServer())
STUN_SERVER = [{"urls": [STUN_STRING]}]

net = load_model(MODEL_LOCAL_PATH)

render_page_header(
    icon="📷",
    title="Realtime Detection",
    subtitle="Detect road damage live using a USB webcam — useful for on-site monitoring with personnel on the ground.",
)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
# Holds (detections, annotated_frame_rgb) only for frames where a pothole was found,
# so the "Report to Authorities" button below can grab the latest one on demand.
pothole_snapshot_queue: "queue.Queue" = queue.Queue(maxsize=1)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:

    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    # Save the results on the queue
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
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    pothole_dets = [d for d in detections if d.label == "Potholes"]
    if pothole_dets:
        if pothole_snapshot_queue.full():
            pothole_snapshot_queue.get_nowait()
        pothole_snapshot_queue.put_nowait((pothole_dets, cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)))

    return av.VideoFrame.from_ndarray(_image, format="bgr24")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.caption("Lower the threshold if damage isn't being detected. Raise it if you're seeing false positives.")

st.write("")
webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
    else:
        st.caption("Start the webcam above to see live predictions here.")

st.divider()
st.markdown('<p class="rs-section-label">Report to Authorities</p>', unsafe_allow_html=True)

if st.button("📸 Capture Latest Pothole & Report", width="stretch", disabled=not webrtc_ctx.state.playing):
    try:
        st.session_state["rt_pothole_snapshot"] = pothole_snapshot_queue.get_nowait()
    except queue.Empty:
        st.session_state["rt_pothole_snapshot"] = None
        st.warning("No pothole detected yet — keep the camera on the road surface and try again.")

snapshot = st.session_state.get("rt_pothole_snapshot")
if snapshot:
    dets, frame_rgb = snapshot
    worst = max(dets, key=lambda d: estimate_severity(d.box, 640, 640)[1])
    severity, severity_pct = estimate_severity(worst.box, 640, 640)

    location = render_location_picker(key_prefix="rt")
    if location:
        lat, lon, address = location
        whatsapp_number, authority_email = render_authority_contact_settings()
        render_report_card(
            key_prefix="rt_report",
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

render_footer()
