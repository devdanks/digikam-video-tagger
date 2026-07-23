from pathlib import Path

from digikam_video_tagger.metadata import ExifToolSidecarWriter


def test_digikam_sidecar_name_keeps_video_extension() -> None:
    assert ExifToolSidecarWriter.sidecar_path(Path("clip.mp4")) == Path("clip.mp4.xmp")
