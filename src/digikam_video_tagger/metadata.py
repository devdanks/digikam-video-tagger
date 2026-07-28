from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from send2trash import send2trash

from .process import run_command
from .tags import validate_tag_path

TAG_FIELDS = ("TagsList", "HierarchicalSubject", "Subject")
TAG_OPTIONS = {
    "TagsList": "XMP-digiKam:TagsList",
    "HierarchicalSubject": "XMP-lr:HierarchicalSubject",
    "Subject": "XMP-dc:Subject",
}
EXIFTOOL_WRITABLE_VIDEO_EXTENSIONS = frozenset({".3g2", ".3gp", ".m4v", ".mov", ".mp4"})
NONPORTABLE_XMP_FIELDS = frozenset({"SourceFile", "XMP-x:XMPToolkit"})


@dataclass(frozen=True)
class MetadataWriteResult:
    media: Path
    added_tags: tuple[str, ...]
    existing_tags: tuple[str, ...]
    recycled_sidecar: Path | None


class MetadataWriteError(RuntimeError):
    """A media write started and may require its source fingerprint to be refreshed."""


class ExifToolMetadataWriter:
    """Merge XMP into writable media and recycle only verified sidecars."""

    def __init__(self, exiftool: Path) -> None:
        self.exiftool = exiftool

    @staticmethod
    def sidecar_path(video: Path) -> Path:
        return Path(f"{video}.xmp")

    @staticmethod
    def supports_video(video: Path) -> bool:
        return video.suffix.casefold() in EXIFTOOL_WRITABLE_VIDEO_EXTENSIONS

    def read_digikam_tags(self, item: Path) -> list[str]:
        return self.read_tag_fields(item)["TagsList"]

    def read_tag_fields(self, item: Path) -> dict[str, list[str]]:
        if not item.exists():
            return {field: [] for field in TAG_FIELDS}
        result = run_command(
            [
                self.exiftool,
                "-j",
                "-XMP-digiKam:TagsList",
                "-XMP-lr:HierarchicalSubject",
                "-XMP-dc:Subject",
                item,
            ],
            timeout=30,
        )
        payload = json.loads(result.stdout or "[]")
        if not payload:
            return {field: [] for field in TAG_FIELDS}
        fields: dict[str, list[str]] = {}
        for field in TAG_FIELDS:
            value = payload[0].get(field, [])
            if value is None:
                fields[field] = []
            elif isinstance(value, str):
                fields[field] = [value]
            elif isinstance(value, list):
                fields[field] = [str(tag) for tag in value]
            else:
                raise ValueError(f"ExifTool returned an invalid {field} value")
        return fields

    def read_xmp_fields(self, item: Path) -> dict[str, object]:
        if not item.exists():
            return {}
        result = run_command(
            [self.exiftool, "-j", "-G1", "-struct", "-XMP:All", item],
            timeout=30,
        )
        payload: object = json.loads(result.stdout or "[]")
        if not isinstance(payload, list) or not payload:
            return {}
        record = payload[0]
        if not isinstance(record, dict):
            raise TypeError("ExifTool returned invalid XMP metadata")
        return {
            str(field): value
            for field, value in record.items()
            if field not in NONPORTABLE_XMP_FIELDS
        }

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        return unique

    @classmethod
    def _desired_tag_fields(
        cls,
        embedded: dict[str, list[str]],
        sidecar: dict[str, list[str]],
        requested_tags: list[str],
    ) -> dict[str, list[str]]:
        desired = {
            field: cls._unique([*embedded[field], *sidecar[field]])
            for field in TAG_FIELDS
        }
        for tag in requested_tags:
            values = {
                "TagsList": tag,
                "HierarchicalSubject": tag.replace("/", "|"),
                "Subject": tag.rsplit("/", 1)[-1],
            }
            for field, value in values.items():
                desired[field] = cls._unique([*desired[field], value])
        return desired

    @staticmethod
    def _missing_tag_fields(
        expected: dict[str, list[str]], actual: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        missing: dict[str, list[str]] = {}
        for field in TAG_FIELDS:
            actual_keys = {value.casefold() for value in actual[field]}
            missing[field] = [
                value
                for value in expected[field]
                if value.casefold() not in actual_keys
            ]
        return missing

    @classmethod
    def _contains_xmp_value(cls, expected: object, actual: object) -> bool:
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and cls._contains_xmp_value(value, actual[key])
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            actual_values = actual if isinstance(actual, list) else [actual]
            return all(
                any(cls._contains_xmp_value(value, item) for item in actual_values)
                for value in expected
            )
        if isinstance(actual, list):
            return any(cls._contains_xmp_value(expected, item) for item in actual)
        return expected == actual

    @classmethod
    def _missing_xmp_fields(
        cls, expected: dict[str, object], actual: dict[str, object]
    ) -> list[str]:
        return [
            field
            for field, value in expected.items()
            if field not in actual or not cls._contains_xmp_value(value, actual[field])
        ]

    @staticmethod
    def _require_writable_video(video: Path, sidecar: Path) -> None:
        if ExifToolMetadataWriter.supports_video(video):
            return
        supported = ", ".join(sorted(EXIFTOOL_WRITABLE_VIDEO_EXTENSIONS))
        retained = f" The sidecar was kept at {sidecar}." if sidecar.exists() else ""
        raise ValueError(
            f"ExifTool cannot embed XMP metadata in {video.suffix or 'this file type'}; "
            f"supported video extensions are {supported}.{retained}"
        )

    def write_tags(self, video: Path, tags: list[str]) -> MetadataWriteResult:
        if not video.is_file():
            raise FileNotFoundError(video)
        original_stat = video.stat()
        sidecar = self.sidecar_path(video)
        if sidecar.exists():
            self._require_writable_video(video, sidecar)
        normalized_tags = [validate_tag_path(tag) for tag in tags]
        embedded_fields = self.read_tag_fields(video)
        sidecar_fields = self.read_tag_fields(sidecar)
        sidecar_xmp = self.read_xmp_fields(sidecar)
        embedded_xmp = self.read_xmp_fields(video) if sidecar_xmp else {}
        tag_options = frozenset(TAG_OPTIONS.values())
        sidecar_xmp_to_import = {
            field: value
            for field, value in sidecar_xmp.items()
            if field not in tag_options and field not in embedded_xmp
        }
        desired_fields = self._desired_tag_fields(
            embedded_fields, sidecar_fields, normalized_tags
        )
        missing_before = self._missing_tag_fields(desired_fields, embedded_fields)
        added_tags = sorted(
            missing_before["TagsList"],
            key=str.casefold,
        )

        needs_write = bool(sidecar_xmp_to_import) or any(missing_before.values())
        if needs_write:
            self._require_writable_video(video, sidecar)
            args: list[str | Path] = [
                self.exiftool,
                "-api",
                "nodups=1",
                "-P",
                "-overwrite_original_in_place",
            ]
            if sidecar_xmp_to_import:
                # Copy only portable XMP fields absent from the media. Existing
                # non-tag values take precedence; the tag lists are merged below.
                args.extend(["-tagsFromFile", sidecar])
                args.extend(f"-{field}" for field in sidecar_xmp_to_import)
            fields_to_write = missing_before
            for field in TAG_FIELDS:
                for value in fields_to_write[field]:
                    args.append(f"-{TAG_OPTIONS[field]}+={value}")
            args.append(video)
            try:
                try:
                    run_command(args, timeout=120)

                    written_fields = self.read_tag_fields(video)
                    missing_after = self._missing_tag_fields(
                        desired_fields, written_fields
                    )
                    missing_values = [
                        f"{field}={value}"
                        for field in TAG_FIELDS
                        for value in missing_after[field]
                    ]
                    if missing_values:
                        raise RuntimeError(
                            "ExifTool did not persist expected media tags: "
                            f"{missing_values}"
                        )
                    if sidecar_xmp_to_import:
                        written_xmp = self.read_xmp_fields(video)
                        missing_xmp = self._missing_xmp_fields(
                            sidecar_xmp_to_import, written_xmp
                        )
                        if missing_xmp:
                            raise RuntimeError(
                                "ExifTool did not persist all sidecar XMP fields: "
                                f"{missing_xmp}"
                            )
                finally:
                    os.utime(
                        video,
                        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                    )
            except Exception as error:
                raise MetadataWriteError(
                    f"Media metadata application failed for {video}: {error}"
                ) from error
        else:
            os.utime(
                video,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

        recycled_sidecar = None
        if sidecar.exists():
            try:
                send2trash(str(sidecar))
            except Exception as error:
                raise MetadataWriteError(
                    f"Media metadata was written but sidecar recycling failed for "
                    f"{video}: {error}"
                ) from error
            recycled_sidecar = sidecar

        return MetadataWriteResult(
            media=video,
            added_tags=tuple(added_tags),
            existing_tags=tuple(embedded_fields["TagsList"]),
            recycled_sidecar=recycled_sidecar,
        )


# Backward compatibility for callers that imported the Alpha-stage class name.
ExifToolSidecarWriter = ExifToolMetadataWriter
