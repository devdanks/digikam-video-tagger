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
