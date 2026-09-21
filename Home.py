import streamlit as st

from sample_utils.ui import (
    inject_base_css,
    render_class_legend,
    render_footer,
    render_hero,
    render_tip,
)

st.set_page_config(
    page_title="RoadShield - Road Damage Detection",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_base_css()

render_hero(
    badge="🛣️ AI-Powered Road Inspection",
    title="Spot road damage before it becomes a hazard",
    description=(
        "RoadShield uses a YOLOv8 deep learning model to automatically detect and "
        "classify cracks and potholes from a webcam feed, a photo, or a video — "
        "so crews can prioritize repairs faster and drivers stay safer."
    ),
)

st.markdown('<p class="rs-section-label">Choose how you want to inspect</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="rs-feature-card glow-indigo">
            <div class="rs-feature-icon">📷</div>
            <h3>Realtime Detection</h3>
            <p>Point a USB webcam at the road and get live detections — ideal for
            on-site monitoring while a vehicle or inspector is moving.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="rs-feature-card glow-violet">
            <div class="rs-feature-icon">🖼️</div>
            <h3>Image Detection</h3>
            <p>Upload a single photo to quickly check a specific stretch of road
            or verify a report someone else submitted.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="rs-feature-card glow-pink">
            <div class="rs-feature-icon">🎬</div>
            <h3>Video Detection</h3>
            <p>Process a recorded drive-through video frame by frame and export
            an annotated copy for your maintenance records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown('<p class="rs-section-label">Damage types it recognizes</p>', unsafe_allow_html=True)
render_class_legend()

st.write("")
render_tip("👈", "Pick a mode from the sidebar to get started.")

with st.expander("About this project"):
    st.markdown(
        """
        RoadShield is built on **YOLOv8-small**, trained on the
        [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://crddc2022.sekilab.global/)
        dataset covering road imagery from Japan and India.

        **License & Citations**
        - Road Damage Dataset from CRDDC2022
        - YOLOv8 by Ultralytics
        """
    )

render_footer()
