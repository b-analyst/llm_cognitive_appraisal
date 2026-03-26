"""Pipeline logging via loguru. Supports capture for the GUI."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

_loguru_logger.remove()
_loguru_logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
    level="INFO",
)
from .config import PIPELINE_ROOT
_logs_dir = PIPELINE_ROOT / "logs"
_logs_dir.mkdir(exist_ok=True)
_loguru_logger.add(
    _logs_dir / "pipeline.log",
    rotation="10 MB",
    retention=3,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
)

_capture_sink_id: int | None = None
_capture_lines: list[str] = []


def get_logger(name: str) -> Any:
    return _loguru_logger.bind(name=name)


def start_capture(session_dir: Path, phase_name: str) -> tuple[int, list[str]]:
    global _capture_sink_id, _capture_lines
    _capture_lines = []
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / f"{phase_name}.log"
    log_path.write_text("", encoding="utf-8")

    def _sink(message: Any) -> None:
        record = message.record
        line = f"{record['time'].strftime('%Y-%m-%d %H:%M:%S')} | {record['level'].name: <8} | {record['message']}"
        _capture_lines.append(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    sink_id = _loguru_logger.add(_sink, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="INFO")
    _capture_sink_id = sink_id
    return sink_id, _capture_lines


def stop_capture(capture_id: int) -> list[str]:
    global _capture_sink_id, _capture_lines
    try:
        _loguru_logger.remove(capture_id)
    except ValueError:
        pass
    if _capture_sink_id == capture_id:
        _capture_sink_id = None
    lines = list(_capture_lines)
    _capture_lines = []
    return lines


def get_capture_lines() -> list[str]:
    return list(_capture_lines)
