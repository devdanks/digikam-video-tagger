"""GPU-assisted video tagging for digiKam."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_opencv_opencl_cache() -> None:
    """Keep OpenCL kernel tuning data between runs for faster startup."""
    if "OPENCV_OCL4DNN_CONFIG_PATH" in os.environ:
        return
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return
    cache_dir = Path(local_app_data) / "digikam-video-tagger" / "opencv-ocl4dnn"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = str(cache_dir)


_configure_opencv_opencl_cache()

__version__ = "0.1.0"
