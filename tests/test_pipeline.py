from pathlib import Path
from types import SimpleNamespace

import digikam_video_tagger.pipeline as pipeline_module
from digikam_video_tagger.pipeline import VideoTaggingPipeline


class FakeSampler:
    def __init__(self, frames: list[Path]) -> None:
        self.frames = frames

    def extract_frames(self, video: Path, **kwargs):
        return (
            SimpleNamespace(codec="h264"),
            self.frames,
            SimpleNamespace(cleanup=lambda: None),
        )


class FakeObjectTagger:
    def detect(self, image):
        return {"cat": 0.9}


def test_pipeline_reports_unreadable_frames_and_counts_them_for_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    good_frame = tmp_path / "good.jpg"
    bad_frame = tmp_path / "bad.jpg"
    good_frame.write_bytes(b"good")
    bad_frame.write_bytes(b"bad")
    monkeypatch.setattr(
        pipeline_module.cv2,
        "imread",
        lambda path, mode: object() if Path(path) == good_frame else None,
    )
    pipeline = VideoTaggingPipeline(
        FakeSampler([good_frame, bad_frame]),
        FakeObjectTagger(),
        None,
        SimpleNamespace(write_tags=lambda *args: None),
        min_object_hits=1,
        min_frame_ratio=0.75,
    )

    result = pipeline.analyze(tmp_path / "video.mp4")

    assert result.unreadable_frames == 1
    assert result.objects == ()
