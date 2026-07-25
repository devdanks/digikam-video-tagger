from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .process import run_command


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_seconds: float
    width: int
    height: int
    codec: str


class FFmpegSampler:
    def __init__(
        self, ffmpeg: Path, ffprobe: Path, *, require_cuda: bool = True
    ) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.require_cuda = require_cuda

    def probe(self, video: Path) -> VideoInfo:
        result = run_command(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height:format=duration",
                "-of",
                "json",
                video,
            ],
            timeout=30,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise ValueError(f"No video stream found in {video}")
        stream = streams[0]
        return VideoInfo(
            video,
            float(data.get("format", {}).get("duration") or 0.0),
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
            str(stream.get("codec_name") or "unknown"),
        )

    def extract_frames(
        self, video: Path, *, sample_seconds: float, max_frames: int, max_dimension: int
    ) -> tuple[VideoInfo, list[Path], tempfile.TemporaryDirectory[str]]:
        if sample_seconds <= 0:
            raise ValueError("sample_seconds must be greater than zero")
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        if max_dimension <= 0:
            raise ValueError("max_dimension must be greater than zero")
        if self.require_cuda:
            self.cuda_smoke_test()
        info = self.probe(video)
        temp_dir = tempfile.TemporaryDirectory(prefix="digikam-video-tags-")
        output_pattern = Path(temp_dir.name) / "%06d.jpg"
        try:
            run_command(
                [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    *(
                        [
                            "-init_hw_device",
                            "cuda=gpu:0",
                            "-filter_hw_device",
                            "gpu",
                            "-hwaccel",
                            "cuda",
                            "-hwaccel_device",
                            "gpu",
                            "-hwaccel_output_format",
                            "cuda",
                        ]
                        if self.require_cuda
                        else []
                    ),
                    "-i",
                    video,
                    "-map",
                    "0:v:0",
                    "-vf",
                    self.sampling_filter(
                        sample_seconds, max_dimension, cuda=self.require_cuda
                    ),
                    "-frames:v",
                    str(max_frames),
                    "-fps_mode",
                    "vfr",
                    "-q:v",
                    "3",
                    output_pattern,
                ],
                timeout=max(60.0, min(3600.0, info.duration_seconds * 2.0)),
            )
            frames = sorted(Path(temp_dir.name).glob("*.jpg"))
            if not frames:
                raise RuntimeError(f"FFmpeg extracted no frames from {video}")
            return info, frames, temp_dir
        except Exception:
            temp_dir.cleanup()
            raise

    @staticmethod
    def sampling_filter(
        sample_seconds: float, max_dimension: int, *, cuda: bool
    ) -> str:
        interval = format(float(sample_seconds), ".9g")
        selection = (
            f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval})'"
        )
        scale = (
            f"{max_dimension}:{max_dimension}:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2:reset_sar=1"
        )
        if cuda:
            return (
                f"{selection},hwupload,scale_cuda={scale},"
                "hwdownload,format=nv12,format=yuvj420p"
            )
        return f"{selection},scale={scale}"

    def cuda_smoke_test(self) -> None:
        run_command(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-init_hw_device",
                "cuda=gpu:0",
                "-filter_hw_device",
                "gpu",
                "-f",
                "lavfi",
                "-i",
                "color=size=64x64:rate=1",
                "-vf",
                "hwupload",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            timeout=30,
        )
