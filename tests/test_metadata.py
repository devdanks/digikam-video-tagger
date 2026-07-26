from pathlib import Path

import pytest

import digikam_video_tagger.metadata as metadata_module
from digikam_video_tagger.metadata import ExifToolSidecarWriter


def test_digikam_sidecar_name_keeps_video_extension() -> None:
    assert ExifToolSidecarWriter.sidecar_path(Path("clip.mp4")) == Path("clip.mp4.xmp")


def test_write_tags_deduplicates_tags_already_embedded_in_video(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls: list[list[object]] = []

    def fake_run_command(args, timeout):
        calls.append(args)
        return type("Result", (), {"stdout": '[{"TagsList": ["People/Alice"]}]'})()

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

    def fake_run_command(args, timeout):
        commands.append(args)
        if "-j" in args:
            item = Path(args[-1])
            payload = (
                "[{}]"
                if item == video
                else '[{"TagsList": ["People/Alice"], "HierarchicalSubject": '
                '["People|Alice"], "Subject": ["Alice"]}]'
            )
            return type("Result", (), {"stdout": payload})()
        output = Path(args[args.index("-o") + 1])
        output.write_text("sidecar", encoding="utf-8")
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    ExifToolSidecarWriter(Path("exiftool")).write_tags(video, ["People/Alice"])

    write_command = next(command for command in commands if "-o" in command)
    assert "-api" in write_command
    assert "nodups=1" in write_command


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


def test_remove_tags_removes_only_requested_owned_values(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("original", encoding="utf-8")
    writer = ExifToolSidecarWriter(Path("exiftool"))

    monkeypatch.setattr(
        writer,
        "read_digikam_tags",
        lambda item: ["People/Unknown/Person_001", "People/Mom"],
    )

    def fake_read_tag_fields(item: Path) -> dict[str, list[str]]:
        # Simulate ExifTool output after removing the placeholder from the temp copy.
        if item.resolve() != sidecar.resolve():
            return {
                "TagsList": ["People/Mom"],
                "HierarchicalSubject": ["People|Mom"],
                "Subject": ["Mom"],
            }
        return {
            "TagsList": ["People/Unknown/Person_001", "People/Mom"],
            "HierarchicalSubject": ["People|Unknown|Person_001", "People|Mom"],
            "Subject": ["Person_001", "Mom"],
        }

    monkeypatch.setattr(writer, "read_tag_fields", fake_read_tag_fields)

    calls: list[list[object]] = []

    def fake_run_command(args, timeout):
        calls.append(args)
        target = Path(args[-1])
        if target.exists():
            target.write_text(" persons", encoding="utf-8")
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    result = writer.remove_tags(video, ["People/Unknown/Person_001"])

    assert result.sidecar == sidecar
    assert result.removed_tags == ("People/Unknown/Person_001",)
    assert result.remaining_tags == ("People/Mom",)
    assert "-XMP-digiKam:TagsList-=People/Unknown/Person_001" in calls[0]
    assert "-XMP-lr:HierarchicalSubject-=People|Unknown|Person_001" in calls[0]
    assert "-XMP-dc:Subject-=Person_001" in calls[0]
    assert not any("People/Mom" in str(value) for value in calls[0])


def test_remove_tags_no_op_for_absent_tags(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("original", encoding="utf-8")
    writer = ExifToolSidecarWriter(Path("exiftool"))

    monkeypatch.setattr(
        writer,
        "read_tag_fields",
        lambda item: {
            "TagsList": ["People/Mom"],
            "HierarchicalSubject": ["People|Mom"],
            "Subject": ["Mom"],
        },
    )
    monkeypatch.setattr(metadata_module, "run_command", lambda args, timeout: None)

    result = writer.remove_tags(video, ["People/Unknown/Person_001"])
    assert result.removed_tags == ()
    assert result.remaining_tags == ("People/Mom",)


def test_remove_tags_raises_when_sidecar_missing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    writer = ExifToolSidecarWriter(Path("exiftool"))

    with pytest.raises(FileNotFoundError, match="sidecar does not exist"):
        writer.remove_tags(video, ["People/Unknown/Person_001"])


def test_remove_tags_rejects_invalid_paths_before_subprocess(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("original", encoding="utf-8")
    writer = ExifToolSidecarWriter(Path("exiftool"))

    with pytest.raises(ValueError, match="tag path segment"):
        writer.remove_tags(video, ["People/A|lice"])


def test_remove_tags_preserves_shared_dc_subject_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = Path(f"{video}.xmp")
    video.write_bytes(b"video")
    sidecar.write_text("original", encoding="utf-8")
    writer = ExifToolSidecarWriter(Path("exiftool"))

    def fake_read_tag_fields(item: Path) -> dict[str, list[str]]:
        if item.resolve() != sidecar.resolve():
            # Temp copy after removal: the tool-owned placeholder is gone, but a
            # different retained tag still owns the shared Person_001 leaf.
            return {
                "TagsList": ["Events/Person_001"],
                "HierarchicalSubject": ["Events|Person_001"],
                "Subject": ["Person_001"],
            }
        return {
            "TagsList": ["People/Unknown/Person_001", "Events/Person_001"],
            "HierarchicalSubject": [
                "People|Unknown|Person_001",
                "Events|Person_001",
            ],
            "Subject": ["Person_001"],
        }

    monkeypatch.setattr(writer, "read_tag_fields", fake_read_tag_fields)

    calls: list[list[object]] = []

    def fake_run_command(args, timeout):
        calls.append(args)
        target = Path(args[-1])
        if target.exists():
            target.write_text(" patched", encoding="utf-8")
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr(metadata_module, "run_command", fake_run_command)

    result = writer.remove_tags(video, ["People/Unknown/Person_001"])

    assert result.removed_tags == ("People/Unknown/Person_001",)
    assert result.remaining_tags == ("Events/Person_001",)
    assert "-XMP-digiKam:TagsList-=People/Unknown/Person_001" in calls[0]
    assert "-XMP-lr:HierarchicalSubject-=People|Unknown|Person_001" in calls[0]
    # The Person_001 leaf is still owned by Events/Person_001, so it must be preserved.
    assert "-XMP-dc:Subject-=Person_001" not in calls[0]
    assert not any(
        isinstance(v, str) and v.startswith("-XMP-dc:Subject-=") for v in calls[0]
    )
