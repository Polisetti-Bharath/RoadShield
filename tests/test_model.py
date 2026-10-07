"""Unit tests for sample_utils/model.py's session-caching behavior.

The actual ultralytics.YOLO() call is mocked out so this test doesn't need
to load real model weights or have `ultralytics`/`torch` installed.
"""

from unittest.mock import MagicMock, patch

import streamlit as st

from sample_utils.model import MODEL_CACHE_KEY, load_model


def setup_function():
    st.session_state.clear()


def test_load_model_instantiates_yolo_on_first_call():
    with patch("sample_utils.model.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance

        net = load_model("fake/path.pt")

        mock_yolo.assert_called_once_with("fake/path.pt")
        assert net is mock_instance
        assert st.session_state[MODEL_CACHE_KEY] is mock_instance


def test_load_model_reuses_cached_instance_on_second_call():
    with patch("sample_utils.model.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()

        first = load_model("fake/path.pt")
        second = load_model("fake/path.pt")

        mock_yolo.assert_called_once()
        assert first is second
