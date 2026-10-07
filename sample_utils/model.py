"""Shared YOLOv8 model loading and detection types for the RoadShield pages."""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import streamlit as st
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes",
]

MODEL_CACHE_KEY = "yolov8smallrdd"


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


def load_model(model_path: Path) -> YOLO:
    """Loads the YOLO model, cached per-session so it's only instantiated once."""
    if MODEL_CACHE_KEY in st.session_state:
        return st.session_state[MODEL_CACHE_KEY]

    logger.info("Loading YOLO model from %s", model_path)
    try:
        net = YOLO(model_path)
    except Exception:
        logger.exception("Failed to load YOLO model from %s", model_path)
        raise
    st.session_state[MODEL_CACHE_KEY] = net
    return net
