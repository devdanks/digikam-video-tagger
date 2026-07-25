from pathlib import Path

from digikam_video_tagger import ffmpeg
from digikam_video_tagger.ffmpeg import FFmpegSampler, VideoInfo


def test_sampling_filter_always_selects_first_frame() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1920, cuda=False)

    assert "isnan(prev_selected_t)" in value
    assert "gte(t-prev_selected_t\\,5)" in value
    assert "fps=" not in value
    assert "scale=1920:1920" in value
    assert "force_original_aspect_ratio=decrease" in value
    assert "force_divisible_by=2" in value
    assert "reset_sar=1" in value


def test_cuda_sampling_filter_keeps_cuda_operations_adjacent() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1920, cuda=True)

    assert "select=" in value
    assert value.index("hwupload") < value.index("scale_cuda")
    assert value.index("scale_cuda") < value.index("hwdownload")
    assert "hwdownload,format=nv12,format=yuvj420p" in value
    assert "isnan(prev_selected_t)" in value


def test_cpu_sampling_filter_constrains_both_dimensions() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1280, cuda=False)

    assert "scale=1280:1280" in value
    assert "force_original_aspect_ratio=decrease" in value
    assert "force_divisible_by=2" in value


def test_cuda_extraction_selects_the_configured_filter_device(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[list[str | Path]] = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        output_pattern = Path(command[-1])
        output_pattern.with_name("000001.jpg").write_bytes(b"frame")

    sampler = FFmpegSampler(Path("ffmpeg.exe"), Path("ffprobe.exe"), require_cuda=True)
    monkeypatch.setattr(sampler, "cuda_smoke_test", lambda: None)
    monkeypatch.setattr(
        sampler,
        "probe",
        lambda video: VideoInfo(video, 1.0, 640, 480, "h264"),
    )
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)

    _, frames, temp_dir = sampler.extract_frames(
        tmp_path / "video.mp4",
        sample_seconds=5.0,
        max_frames=1,
        max_dimension=1920,
    )
    temp_dir.cleanup()

    assert frames
    assert "-init_hw_device" in commands[0]
    assert commands[0][commands[0].index("-init_hw_device") + 1] == "cuda=gpu:0"
    assert "-filter_hw_device" in commands[0]
    assert commands[0][commands[0].index("-filter_hw_device") + 1] == "gpu"
