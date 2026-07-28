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
    embed_args = parser.parse_args(["embed", str(tmp_path)])

    assert prepare_args.recursive is True
    assert finalize_args.recursive is True
    assert status_args.recursive is True
    assert embed_args.recursive is True
    assert embed_args.apply is False
    assert status_args.summary_only is True
    assert status_args.apply is False


def test_backend_flags_can_be_controlled_independently(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["prepare", str(tmp_path), "--no-ffmpeg-cuda", "--opencl"]
    )

    assert args.ffmpeg_cuda is False
    assert args.opencl is True


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


def test_finalize_apply_reports_embedded_media_and_recycled_sidecar(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "reviewed.mp4"
    sidecar = Path(f"{source}.xmp")
    frame = tmp_path / "frame.jpg"
    source.write_bytes(b"video")
    frame.write_bytes(b"frame")
    removed = []
    recorded = []
    written = []
    job = SimpleNamespace(
        source_path=source,
        job_dir=tmp_path,
        frame_paths=[frame],
        frames=(object(),),
        source_is_unchanged=lambda: True,
        remove_generated_files=lambda: removed.append(frame) or [frame],
    )
    face = SimpleNamespace(
        frame=frame,
        image_id=1,
        person_tag_paths=("People/Alice",),
    )
    catalog = SimpleNamespace(confirmed_faces_for_frames=lambda paths: [face])
    writer = SimpleNamespace(
        write_tags=lambda media, tags: (
            written.append((media, tags))
            or SimpleNamespace(
                media=media,
                added_tags=("People/Alice",),
                recycled_sidecar=sidecar,
            )
        )
    )
    monkeypatch.setattr(cli, "discover_jobs", lambda staging, sources: [job])
    monkeypatch.setattr(cli, "DigiKamCatalog", lambda config: catalog)
    monkeypatch.setattr(cli, "ExifToolMetadataWriter", lambda exiftool: writer)
    monkeypatch.setattr(
        cli,
        "mark_job_completed",
        lambda completed_job, people, retained_sidecar: recorded.append(
            (completed_job, people, retained_sidecar)
        ),
    )
    monkeypatch.setattr(cli, "_completed_count", lambda staging, sources: 1)

    args = build_parser().parse_args(
        ["finalize", "--staging-dir", str(tmp_path), "--apply"]
    )

    assert args.handler(args) == 0
    assert written == [(source, ["People/Alice"])]
    assert recorded == [(job, ["People/Alice"], None)]
    assert removed == [frame]
    output = capsys.readouterr().out
    assert f"media={source}" in output
    assert f"sidecar_recycled={sidecar}" in output


def test_finalize_refreshes_job_fingerprint_after_retryable_metadata_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "reviewed.mp4"
    frame = tmp_path / "frame.jpg"
    source.write_bytes(b"video")
    frame.write_bytes(b"frame")
    refreshed = []
    removed = []
    job = SimpleNamespace(
        source_path=source,
        job_dir=tmp_path,
        frame_paths=[frame],
        frames=(object(),),
        source_is_unchanged=lambda: True,
        remove_generated_files=lambda: removed.append(frame),
    )
    face = SimpleNamespace(
        frame=frame,
        image_id=1,
        person_tag_paths=("People/Alice",),
    )
    catalog = SimpleNamespace(confirmed_faces_for_frames=lambda paths: [face])
    writer = SimpleNamespace(
        write_tags=lambda media, tags: (_ for _ in ()).throw(
            cli.MetadataWriteError("Recycle Bin unavailable")
        )
    )
    monkeypatch.setattr(cli, "discover_jobs", lambda staging, sources: [job])
    monkeypatch.setattr(cli, "DigiKamCatalog", lambda config: catalog)
    monkeypatch.setattr(cli, "ExifToolMetadataWriter", lambda exiftool: writer)
    monkeypatch.setattr(
        cli,
        "refresh_job_source_fingerprint",
        lambda failed_job: refreshed.append(failed_job),
    )
    monkeypatch.setattr(cli, "_completed_count", lambda staging, sources: 0)

    args = build_parser().parse_args(
        ["finalize", "--staging-dir", str(tmp_path), "--apply"]
    )

    assert args.handler(args) == 1
    assert refreshed == [job]
    assert removed == []


def test_embed_apply_consumes_existing_sidecar(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = (tmp_path / "legacy.mp4").resolve()
    sidecar = Path(f"{source}.xmp")
    source.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    written = []
    writer = SimpleNamespace(
        sidecar_path=lambda media: Path(f"{media}.xmp"),
        supports_video=lambda media: True,
        write_tags=lambda media, tags: (
            written.append((media, tags))
            or SimpleNamespace(
                media=media,
                added_tags=("People/Alice",),
                recycled_sidecar=sidecar,
            )
        ),
    )
    monkeypatch.setattr(cli, "ExifToolMetadataWriter", lambda exiftool: writer)

    args = build_parser().parse_args(["embed", str(tmp_path), "--apply"])

    assert args.handler(args) == 0
    assert written == [(source, [])]
    output = capsys.readouterr().out
    assert "[1/1 EMBEDDED]" in output
    assert f"media={source}" in output
    assert f"sidecar_recycled={sidecar}" in output


def test_embed_preview_identifies_unsupported_media_without_writing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = (tmp_path / "legacy.mkv").resolve()
    sidecar = Path(f"{source}.xmp")
    source.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    writer = SimpleNamespace(
        sidecar_path=lambda media: Path(f"{media}.xmp"),
        supports_video=lambda media: False,
        write_tags=lambda media, tags: pytest.fail("preview must not write metadata"),
    )
    monkeypatch.setattr(cli, "ExifToolMetadataWriter", lambda exiftool: writer)

    args = build_parser().parse_args(["embed", str(tmp_path)])

    assert args.handler(args) == 0
    output = capsys.readouterr().out
    assert "[1/1 UNSUPPORTED]" in output
    assert "1 unsupported" in output
    assert sidecar.exists()
