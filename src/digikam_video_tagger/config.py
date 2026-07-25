from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _environment_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _command_path(*names: str) -> Path | None:
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
DISCOVERED_FFMPEG = _command_path("ffmpeg.exe", "ffmpeg")
DISCOVERED_EXIFTOOL = _command_path("exiftool.exe", "exiftool")

DEFAULT_FFMPEG_DIR = _environment_path(
    "DIGIKAM_VIDEO_TAGGER_FFMPEG_DIR",
    DISCOVERED_FFMPEG.parent if DISCOVERED_FFMPEG else Path("."),
)
DEFAULT_EXIFTOOL = _environment_path(
    "DIGIKAM_VIDEO_TAGGER_EXIFTOOL",
    DISCOVERED_EXIFTOOL if DISCOVERED_EXIFTOOL else Path("exiftool.exe"),
)
DEFAULT_STAGING_DIR = _environment_path(
    "DIGIKAM_VIDEO_TAGGER_STAGING_DIR",
    Path.home() / "Pictures" / "_digikam_video_faces",
)
DEFAULT_DIGIKAM_CONFIG = _environment_path(
    "DIGIKAM_VIDEO_TAGGER_DIGIKAM_CONFIG",
    LOCAL_APP_DATA / "digikamrc",
)
DEFAULT_MODEL_DIR = _environment_path(
    "DIGIKAM_VIDEO_TAGGER_MODEL_DIR",
    LOCAL_APP_DATA / "digikam" / "facesengine",
)
DEFAULT_DB_HOST = os.environ.get("DIGIKAM_VIDEO_TAGGER_DB_HOST", "127.0.0.1")
DEFAULT_DB_PORT = _environment_int("DIGIKAM_VIDEO_TAGGER_DB_PORT", 3307)
DEFAULT_DB_USER = os.environ.get("DIGIKAM_VIDEO_TAGGER_DB_USER", "root")
DEFAULT_DB_PASSWORD = os.environ.get("DIGIKAM_VIDEO_TAGGER_DB_PASSWORD", "")
DEFAULT_DB_NAME = os.environ.get("DIGIKAM_VIDEO_TAGGER_DB_NAME", "digikam")


def read_kconfig_boolean(path: Path, section: str, key: str) -> bool | None:
    """Read one boolean from a KDE-style INI file without rewriting it."""
    if not path.is_file():
        return None

    wanted_section = section.casefold()
    wanted_key = key.casefold()
    current_section = ""
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().casefold()
            continue
        if current_section != wanted_section or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip().casefold() != wanted_key:
            continue
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        return None
    return None


def digikam_sidecar_reading_enabled(path: Path) -> bool | None:
    """Read digiKam's sidecar-reading option across supported config key names."""
    for key in ("Use XMP Sidecar For Reading", "UseXMPSidecar4Reading"):
        value = read_kconfig_boolean(path, "Metadata Settings", key)
        if value is not None:
            return value
    return None


VIDEO_EXTENSIONS = frozenset(
    {
        ".3g2",
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".m2ts",
        ".m2v",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpe",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".vob",
        ".webm",
        ".wmv",
    }
)


@dataclass(frozen=True)
class ToolPaths:
    ffmpeg_dir: Path = DEFAULT_FFMPEG_DIR
    exiftool: Path = DEFAULT_EXIFTOOL
    model_dir: Path = DEFAULT_MODEL_DIR

    @property
    def ffmpeg(self) -> Path:
        return self.ffmpeg_dir / "ffmpeg.exe"

    @property
    def ffprobe(self) -> Path:
        return self.ffmpeg_dir / "ffprobe.exe"

    @property
    def yunet(self) -> Path:
        return self.model_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def sface(self) -> Path:
        return self.model_dir / "face_recognition_sface_2021dec.onnx"

    @property
    def yolo_nano(self) -> Path:
        return self.model_dir / "yolo11n.onnx"

    @property
    def yolo_xlarge(self) -> Path:
        return self.model_dir / "yolo11x.onnx"

    @property
    def coco_names(self) -> Path:
        return self.model_dir / "coco.names"


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = DEFAULT_DB_HOST
    port: int = DEFAULT_DB_PORT
    user: str = DEFAULT_DB_USER
    password: str = DEFAULT_DB_PASSWORD
    database: str = DEFAULT_DB_NAME
