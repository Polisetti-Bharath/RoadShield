"""Unit tests for sample_utils/logging_config.py.

pytest's own `caplog` fixture attaches a handler to the root logger for every
test, so these assert on calls to `logging.basicConfig` rather than on
`root.handlers` being empty.
"""

from unittest.mock import patch

from sample_utils.logging_config import configure_logging


def test_configure_logging_calls_basicConfig_when_root_has_no_handlers():
    with patch("sample_utils.logging_config.logging.getLogger") as mock_get_logger, \
         patch("sample_utils.logging_config.logging.basicConfig") as mock_basic_config:
        mock_get_logger.return_value.handlers = []

        configure_logging()

        mock_basic_config.assert_called_once()


def test_configure_logging_is_a_noop_when_root_already_has_a_handler():
    with patch("sample_utils.logging_config.logging.getLogger") as mock_get_logger, \
         patch("sample_utils.logging_config.logging.basicConfig") as mock_basic_config:
        mock_get_logger.return_value.handlers = ["existing-handler"]

        configure_logging()

        mock_basic_config.assert_not_called()


def test_configure_logging_reads_log_level_env_var(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    with patch("sample_utils.logging_config.logging.getLogger") as mock_get_logger, \
         patch("sample_utils.logging_config.logging.basicConfig") as mock_basic_config:
        mock_get_logger.return_value.handlers = []

        configure_logging()

        assert mock_basic_config.call_args.kwargs["level"] == "DEBUG"


def test_configure_logging_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    with patch("sample_utils.logging_config.logging.getLogger") as mock_get_logger, \
         patch("sample_utils.logging_config.logging.basicConfig") as mock_basic_config:
        mock_get_logger.return_value.handlers = []

        configure_logging()

        assert mock_basic_config.call_args.kwargs["level"] == "INFO"
