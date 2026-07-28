import os
from pathlib import Path

import pytest

import digikam_video_tagger.metadata as metadata_module
from digikam_video_tagger.metadata import ExifToolSidecarWriter


def test_digikam_sidecar_name_keeps_video_extension() -> None:
    assert ExifToolSidecarWriter.sidecar_path(Path("clip.mp4")) == Path("clip.mp4.xmp")


def test_read_xmp_fields_ignores_exiftool_serialization_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    sidecar = tmp_path / "clip.mp4.xmp"
    sidecar.write_text("xmp", encoding="utf-8")
    monkeypatch.setattr(
        metadata_module,
        "run_command",
        lambda args, timeout: type(
            "Result",
            (),
            {
                "stdout": (
                    '[{"SourceFile": "clip.mp4.xmp", '
                    '"XMP-x:XMPToolkit": "XMP Core 6.0.0", '
                    '"XMP-dc:Title": "Keep me"}]'
                )
            },
        )(),
    )

    fields = ExifToolSidecarWriter(Path("exiftool")).read_xmp_fields(sidecar)

    assert fields == {"XMP-dc:Title": "Keep me"}


def test_write_tags_embeds_sidecar_tags_then_recycles_verified_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    embedded = False
    events: list[str] = []

    def fake_run_command(args, timeout):
        nonlocal embedded
        if "-j" in args:
            item = Path(args[-1])
            if item == sidecar or embedded:
                if item == video and embedded:
                    events.append("verify")
                payload = (
                    '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                    '["People|Alice"], "Subject": ["Alice"]}]'
                )
            else:
                payload = "[{}]"
            return type("Result", (), {"stdout": payload})()
        events.append("write")
        embedded = True
        return type("Result", (), {"stdout": ""})()

    def fake_send2trash(path: str) -> None:
        events.append("trash")
        Path(path).unlink()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(metadata_module, "send2trash", fake_send2trash, raising=False)

    result = ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert events[0] == "write"
    assert events[-1] == "trash"
    assert events[1:-1] and set(events[1:-1]) == {"verify"}
    assert not sidecar.exists()
    assert result.media == video
    assert result.added_tags == ("People/Alice",)
    assert result.recycled_sidecar == sidecar


def test_write_tags_deduplicates_tags_already_embedded_in_video(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls: list[list[object]] = []

    def fake_run_command(args, timeout):
        calls.append(args)
        return type(
            "Result",
            (),
            {
                "stdout": (
                    '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                    '["People|Alice"], "Subject": ["Alice"]}]'
                )
            },
        )()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    result = ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert result.added_tags == ()
    assert len(calls) == 1


def test_write_tags_requests_exiftool_deduplication(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    commands: list[list[object]] = []
    embedded = False

    def fake_run_command(args, timeout):
        nonlocal embedded
        commands.append(args)
        if "-j" in args:
            payload = (
                '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
                if embedded
                else "[{}]"
            )
            return type("Result", (), {"stdout": payload})()
        embedded = True
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    write_command = next(command for command in commands if "-j" not in command)
    assert "-api" in write_command
    assert "nodups=1" in write_command
    assert "-P" in write_command
    assert "-overwrite_original_in_place" in write_command
    assert "-o" not in write_command
    assert Path(write_command[-1]) == video


def test_write_tags_restores_exact_media_timestamps(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    os.utime(video, ns=(1_700_000_000_123_456_700, 1_700_000_001_765_432_100))
    original_stat = video.stat()
    embedded = False

    def fake_run_command(args, timeout):
        nonlocal embedded
        if "-j" in args:
            if not embedded:
                os.utime(
                    video,
                    ns=(
                        original_stat.st_atime_ns + 5_000_000_000,
                        original_stat.st_mtime_ns,
                    ),
                )
            payload = (
                '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
                if embedded
                else "[{}]"
            )
            return type("Result", (), {"stdout": payload})()
        embedded = True
        os.utime(
            video,
            ns=(
                original_stat.st_atime_ns + 10_000_000_000,
                original_stat.st_mtime_ns + 10_000_000_000,
            ),
        )
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    written_stat = video.stat()
    assert written_stat.st_atime_ns == original_stat.st_atime_ns
    assert written_stat.st_mtime_ns == original_stat.st_mtime_ns


def test_write_tags_adds_only_values_missing_from_the_embedded_tag_union(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    commands: list[list[object]] = []
    embedded = False

    def fake_run_command(args, timeout):
        nonlocal embedded
        commands.append(args)
        if "-j" in args:
            if "-G1" in args:
                payload = "[{}]"
            elif embedded:
                payload = (
                    '[{"TagsList": ["People/Bob", "People/Alice", "People/Carol"], '
                    '"HierarchicalSubject": ["People|Bob", "People|Alice", '
                    '"People|Carol"], "Subject": ["Bob", "Alice", "Carol"]}]'
                )
            elif Path(args[-1]) == sidecar:
                payload = (
                    '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                    '["People|Alice"], "Subject": ["Alice"]}]'
                )
            else:
                payload = (
                    '[{"TagsList": ["People/Bob"], "HierarchicalSubject": '
                    '["People|Bob"], "Subject": ["Bob"]}]'
                )
            return type("Result", (), {"stdout": payload})()
        embedded = True
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(metadata_module, "send2trash", lambda path: sidecar.unlink())

    ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Carol"])

    write_command = next(command for command in commands if "-j" not in command)
    assert "-tagsFromFile" not in write_command
    assert "-XMP-digiKam:TagsList=People/Bob" not in write_command
    assert "-XMP-digiKam:TagsList+=People/Bob" not in write_command
    assert "-XMP-digiKam:TagsList+=People/Alice" in write_command
    assert "-XMP-digiKam:TagsList+=People/Carol" in write_command


def test_write_tags_preserves_existing_non_tag_xmp_values(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    commands: list[list[object]] = []
    embedded = False

    def fake_run_command(args, timeout):
        nonlocal embedded
        commands.append(args)
        item = Path(args[-1])
        if "-j" in args:
            if "-G1" in args:
                title = "Sidecar title" if item == sidecar else "Embedded title"
                return type(
                    "Result", (), {"stdout": f'[{{"XMP-dc:Title": "{title}"}}]'}
                )()
            payload = (
                '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
                if item == sidecar or embedded
                else "[{}]"
            )
            return type("Result", (), {"stdout": payload})()
        embedded = True
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(metadata_module, "send2trash", lambda path: sidecar.unlink())

    ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    write_command = next(command for command in commands if "-j" not in command)
    assert "-tagsFromFile" not in write_command
    assert sidecar.exists() is False


def test_write_tags_keeps_sidecar_when_media_tag_verification_fails(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    write_finished = False
    recycled: list[str] = []

    def fake_run_command(args, timeout):
        nonlocal write_finished
        if "-j" in args:
            item = Path(args[-1])
            payload = (
                '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
                if item == sidecar
                else "[{}]"
            )
            return type("Result", (), {"stdout": payload})()
        write_finished = True
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(metadata_module, "send2trash", recycled.append)

    with pytest.raises(RuntimeError, match="did not persist expected media tags"):
        ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert write_finished is True
    assert sidecar.exists()
    assert recycled == []


def test_write_tags_reports_post_write_recycle_failure_as_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    embedded = False

    def fake_run_command(args, timeout):
        nonlocal embedded
        if "-j" in args:
            payload = (
                '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
                if Path(args[-1]) == sidecar or embedded
                else "[{}]"
            )
            return type("Result", (), {"stdout": payload})()
        embedded = True
        video.write_bytes(b"video with embedded metadata")
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(
        metadata_module,
        "send2trash",
        lambda path: (_ for _ in ()).throw(OSError("Recycle Bin unavailable")),
    )

    with pytest.raises(
        metadata_module.MetadataWriteError,
        match="Recycle Bin unavailable",
    ):
        ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert sidecar.exists()
    assert video.read_bytes() == b"video with embedded metadata"


def test_write_tags_keeps_sidecar_for_video_type_exiftool_cannot_write(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mkv"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("sidecar", encoding="utf-8")
    commands: list[list[object]] = []
    recycled: list[str] = []

    def fake_run_command(args, timeout):
        commands.append(args)
        payload = (
            '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
            '["People|Alice"], "Subject": ["Alice"]}]'
            if Path(args[-1]) == sidecar
            else "[{}]"
        )
        return type("Result", (), {"stdout": payload})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)
    monkeypatch.setattr(metadata_module, "send2trash", recycled.append)

    with pytest.raises(ValueError, match="cannot embed XMP metadata in .mkv"):
        ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert commands == []
    assert sidecar.exists()
    assert recycled == []


def test_write_tags_rejects_unsupported_video_without_an_existing_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"video")
    commands: list[list[object]] = []

    def fake_run_command(args, timeout):
        commands.append(args)
        return type("Result", (), {"stdout": "[{}]"})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    with pytest.raises(ValueError, match="cannot embed XMP metadata in .mkv"):
        ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    assert commands and all("-j" in command for command in commands)


def test_read_tag_fields_treats_null_as_an_empty_tag_list(
    tmp_path: Path, monkeypatch
) -> None:
    item = tmp_path / "clip.mp4.xmp"
    item.write_text("sidecar", encoding="utf-8")

    def fake_run_command(args, timeout):
        return type(
            "Result",
            (),
            {
                "stdout": (
                    '[{"TagsList": null, "HierarchicalSubject": null, "Subject": null}]'
                )
            },
        )()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    assert ExifToolSidecarWriter(Path("exiftool")).read_tag_fields(item) == {
        "TagsList": [],
        "HierarchicalSubject": [],
        "Subject": [],
    }


def test_write_tags_rejects_reserved_hierarchy_characters(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="'\\|'"):
        ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/A|lice"])
