from pathlib import Path
from types import SimpleNamespace

import pytest

from digikam_video_tagger import cli
from digikam_video_tagger.cli import _discover_videos, build_parser


def test_video_discovery_is_recursive_and_filters_extensions(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "first.MP4").write_bytes(b"")
    (nested / "second.mkv").write_bytes(b"")
    (nested / "ignore.jpg").write_bytes(b"")

    videos = _discover_videos([tmp_path], recursive=True)

    assert videos == sorted(
        [(tmp_path / "first.MP4").resolve(), (nested / "second.mkv").resolve()],
        key=lambda item: str(item).casefold(),
    )


def test_folder_commands_are_recursive_by_default(tmp_path: Path) -> None:
    parser = build_parser()

    prepare_args = parser.parse_args(["prepare", str(tmp_path)])
    finalize_args = parser.parse_args(["finalize", str(tmp_path)])
    status_args = parser.parse_args(["status", str(tmp_path)])
    autofinalize_args = parser.parse_args(["autofinalize", str(tmp_path)])

    assert prepare_args.recursive is True
    assert finalize_args.recursive is True
    assert status_args.recursive is True
    assert autofinalize_args.recursive is True
    assert status_args.summary_only is True
    assert status_args.apply is False


def test_backend_flags_can_be_controlled_independently(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["prepare", str(tmp_path), "--no-ffmpeg-cuda", "--opencl"]
    )

    assert args.ffmpeg_cuda is False
    assert args.opencl is True


def test_autofinalize_requires_a_source_path() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["autofinalize"])
    assert error.value.code == 2


def test_autofinalize_handler_outputs_json_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    class FakeService:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(
            self, videos: list[Path], *, apply: bool, reprocess_completed: bool
        ) -> tuple[list[object], object]:
            result = SimpleNamespace(
                source_video=videos[0],
                job_id="abc",
                frame_count=1,
                unreadable_frames=0,
                face_frames=1,
                known_people=("People/Mom",),
                placeholder_people=(),
                proposed_replacements=(),
                completed=True,
                applied=apply,
                sidecar=None,
                removed_proxy_files=0,
                error=None,
            )
            summary = SimpleNamespace(
                videos=1,
                applied=1 if apply else 0,
                completed=1,
                known_people=1,
                clustered_people=0,
                resolved_people=0,
                failed=0,
            )
            return [result], summary

    class FakeClusterStore:
        @classmethod
        def empty(cls, **kwargs: object) -> object:
            return SimpleNamespace(
                unknown_root="People/Unknown",
                store_id="store",
                clusters={},
                save=lambda path: None,
            )

        @classmethod
        def load(cls, path: Path, **kwargs: object) -> object:
            return cls.empty()

    monkeypatch.setattr(cli, "AutoFinalizeService", FakeService)
    monkeypatch.setattr(cli, "FaceClusterStore", FakeClusterStore)
    monkeypatch.setattr(cli, "_model_fingerprint", lambda path: "x")
    monkeypatch.setattr(cli, "select_opencv_target", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli, "DigiKamFaceGallery", lambda config: SimpleNamespace(load=list)
    )
    monkeypatch.setattr(cli, "FaceTagger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, "FFmpegSampler", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli, "ExifToolSidecarWriter", lambda *args, **kwargs: SimpleNamespace()
    )

    args = build_parser().parse_args(["autofinalize", str(tmp_path), "--json"])
    assert args.handler(args) == 0
    output = capsys.readouterr().out
    assert '"type": "summary"' in output
    assert '"command": "autofinalize"' in output
    assert '"applied": 0' in output


def test_status_is_successful_when_no_active_jobs(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["status", "--staging-dir", str(tmp_path)])

    assert args.handler(args) == 0
    assert "0 active job(s)" in capsys.readouterr().out


def test_finalize_apply_fails_when_proxy_frames_are_unscanned(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "pending.mp4"
    frame = tmp_path / "frame.jpg"
    source.write_bytes(b"video")
    frame.write_bytes(b"frame")
    job = SimpleNamespace(
        source_path=source,
        job_dir=tmp_path,
        frame_paths=[frame],
        frames=(object(),),
        source_is_unchanged=lambda: True,
    )
    face = SimpleNamespace(frame=frame, image_id=None, person_tag_paths=())
    catalog = SimpleNamespace(confirmed_faces_for_frames=lambda paths: [face])
    monkeypatch.setattr(cli, "discover_jobs", lambda staging, sources: [job])
    monkeypatch.setattr(cli, "DigiKamCatalog", lambda config: catalog)

    args = build_parser().parse_args(
        ["finalize", "--staging-dir", str(tmp_path), "--apply"]
    )

    assert args.handler(args) == 1
    output = capsys.readouterr().err
    assert "Cannot apply an incomplete job" in output


def test_finalize_apply_completes_scanned_job_without_people(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "reviewed.mp4"
    frame = tmp_path / "frame.jpg"
    source.write_bytes(b"video")
    frame.write_bytes(b"frame")
    removed = []
    recorded = []
    job = SimpleNamespace(
        source_path=source,
        job_dir=tmp_path,
        frame_paths=[frame],
        frames=(object(),),
        source_is_unchanged=lambda: True,
        remove_generated_files=lambda: removed.append(frame) or [frame],
    )
    face = SimpleNamespace(frame=frame, image_id=1, person_tag_paths=())
    catalog = SimpleNamespace(confirmed_faces_for_frames=lambda paths: [face])
    monkeypatch.setattr(cli, "discover_jobs", lambda staging, sources: [job])
    monkeypatch.setattr(cli, "DigiKamCatalog", lambda config: catalog)
    monkeypatch.setattr(
        cli,
        "mark_job_completed",
        lambda completed_job, people, sidecar: recorded.append(
            (completed_job, people, sidecar)
        ),
    )
    monkeypatch.setattr(cli, "_completed_count", lambda staging, sources: 1)

    args = build_parser().parse_args(
        ["finalize", "--staging-dir", str(tmp_path), "--apply"]
    )

    assert args.handler(args) == 0
    assert recorded == [(job, [], None)]
    assert removed == [frame]
    output = capsys.readouterr().out
    assert "APPLIED-NO-PEOPLE" in output
    assert "1 applied" in output
