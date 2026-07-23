from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .process import CommandError, run_command


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_seconds: float
    width: int
    height: int
    codec: str


class FFmpegSampler:
    def __init__(self, ffmpeg: Path, ffprobe: Path, *, require_cuda: bool = True) -> None:
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
        duration = float(data.get("format", {}).get("duration") or 0.0)
        return VideoInfo(
            path=video,
            duration_seconds=duration,
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            codec=str(stream.get("codec_name") or "unknown"),
        )

    def extract_frames(
        self,
        video: Path,
        *,
        sample_seconds: float,
        max_frames: int,
        max_dimension: int,
    ) -> tuple[VideoInfo, list[Path], tempfile.TemporaryDirectory[str]]:
        if sample_seconds <= 0:
            raise ValueError("sample_seconds must be greater than zero")
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")

        info = self.probe(video)
        temp_dir = tempfile.TemporaryDirectory(prefix="digikam-video-tags-")
        output_pattern = Path(temp_dir.name) / "%06d.jpg"
        vf_cuda = self.sampling_filter(sample_seconds, max_dimension, cuda=True)
        cuda_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        args = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            *cuda_args,
            "-i",
            video,
            "-map",
            "0:v:0",
            "-vf",
            vf_cuda,
            "-frames:v",
            str(max_frames),
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            output_pattern,
        ]

        try:
            run_command(args, timeout=max(60.0, min(3600.0, info.duration_seconds * 2.0)))
        except CommandError:
            temp_dir.cleanup()
            if self.require_cuda:
                raise
            temp_dir = tempfile.TemporaryDirectory(prefix="digikam-video-tags-")
            output_pattern = Path(temp_dir.name) / "%06d.jpg"
            cpu_filter = self.sampling_filter(sample_seconds, max_dimension, cuda=False)
            run_command(
                [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video,
                    "-map",
                    "0:v:0",
                    "-vf",
                    cpu_filter,
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
            temp_dir.cleanup()
            raise RuntimeError(f"FFmpeg extracted no frames from {video}")
        return info, frames, temp_dir

    @staticmethod
    def sampling_filter(sample_seconds: float, max_dimension: int, *, cuda: bool) -> str:
        """Select the first frame, then one frame per interval.

        FFmpeg's fps filter centers its first sampling window. For clips shorter
        than half the interval that can produce no output at all. The select
        expression is first-frame-safe and works with variable-frame-rate video.
        """
        interval = format(float(sample_seconds), ".9g")
        selection = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval})'"
        scale = (
            f"{max_dimension}:{max_dimension}:"
            "force_original_aspect_ratio=decrease:"
            "force_divisible_by=2:reset_sar=1"
        )
        if cuda:
            return (
                f"{selection},"
                f"scale_cuda={scale},"
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
