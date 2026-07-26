from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .process import run_command
from .tags import validate_tag_path


@dataclass(frozen=True)
class MetadataWriteResult:
    sidecar: Path
    added_tags: tuple[str, ...]
    existing_tags: tuple[str, ...]


@dataclass(frozen=True)
class MetadataRemovalResult:
    sidecar: Path
    removed_tags: tuple[str, ...]
    remaining_tags: tuple[str, ...]


class ExifToolSidecarWriter:
    def __init__(self, exiftool: Path) -> None:
        self.exiftool = exiftool

    @staticmethod
    def sidecar_path(video: Path) -> Path:
        return Path(f"{video}.xmp")

    def read_digikam_tags(self, item: Path) -> list[str]:
        return self.read_tag_fields(item)["TagsList"]

    def read_tag_fields(self, item: Path) -> dict[str, list[str]]:
        if not item.exists():
            return {"TagsList": [], "HierarchicalSubject": [], "Subject": []}
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
            return {"TagsList": [], "HierarchicalSubject": [], "Subject": []}
        fields: dict[str, list[str]] = {}
        for field in ("TagsList", "HierarchicalSubject", "Subject"):
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

    def write_tags(self, video: Path, tags: list[str]) -> MetadataWriteResult:
        sidecar = self.sidecar_path(video)
        normalized_tags = [validate_tag_path(tag) for tag in tags]
        existing_tags = self.read_digikam_tags(sidecar)
        embedded_tags = self.read_digikam_tags(video)
        existing_keys = {tag.casefold() for tag in [*existing_tags, *embedded_tags]}
        new_tags = sorted(
            {
                tag.strip()
                for tag in normalized_tags
                if tag.strip() and tag.strip().casefold() not in existing_keys
            },
            key=str.casefold,
        )
        if not new_tags:
            return MetadataWriteResult(sidecar, (), tuple(existing_tags))

        sidecar.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".xmp", dir=sidecar.parent
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            if sidecar.exists():
                shutil.copy2(sidecar, temp_path)
                source = temp_path
                args: list[str | Path] = [
                    self.exiftool,
                    "-m",
                    "-api",
                    "nodups=1",
                    "-overwrite_original",
                ]
            else:
                temp_path.unlink()
                source = video
                args = [
                    self.exiftool,
                    "-m",
                    "-api",
                    "nodups=1",
                    "-overwrite_original",
                    "-o",
                    temp_path,
                ]

            for tag in new_tags:
                args.extend(
                    [
                        f"-XMP-digiKam:TagsList+={tag}",
                        f"-XMP-lr:HierarchicalSubject+={tag.replace('/', '|')}",
                        f"-XMP-dc:Subject+={tag.rsplit('/', 1)[-1]}",
                    ]
                )
            args.append(source)
            run_command(args, timeout=120)

            written = self.read_tag_fields(temp_path)
            tags_list = {value.casefold() for value in written["TagsList"]}
            hierarchical = {
                value.casefold() for value in written["HierarchicalSubject"]
            }
            subjects = {value.casefold() for value in written["Subject"]}
            missing = [
                tag
                for tag in new_tags
                if tag.casefold() not in tags_list
                or tag.replace("/", "|").casefold() not in hierarchical
                or tag.rsplit("/", 1)[-1].casefold() not in subjects
            ]
            if missing:
                raise RuntimeError(f"ExifTool did not persist expected tags: {missing}")
            os.replace(temp_path, sidecar)
        finally:
            temp_path.unlink(missing_ok=True)

        return MetadataWriteResult(sidecar, tuple(new_tags), tuple(existing_tags))

    def remove_tags(self, video: Path, tags: list[str]) -> MetadataRemovalResult:
        sidecar = self.sidecar_path(video)
        if not sidecar.exists():
            raise FileNotFoundError(f"sidecar does not exist: {sidecar}")

        requested = []
        seen: set[str] = set()
        for tag in tags:
            validated = validate_tag_path(tag)
            if validated.casefold() not in seen:
                requested.append(validated)
                seen.add(validated.casefold())

        fields = self.read_tag_fields(sidecar)
        current_tags_list = set(fields["TagsList"])
        to_remove = [tag for tag in requested if tag in current_tags_list]
        if not to_remove:
            return MetadataRemovalResult(
                sidecar, (), tuple(sorted(fields["TagsList"], key=str.casefold))
            )

        # dc:Subject stores flat leaf names shared across every hierarchical tag.
        # Only retract a leaf when no remaining tag still references it, so removing
        # a tool-owned placeholder never strips a Subject value owned by another tag.
        remaining_tags_list = current_tags_list - set(to_remove)
        referenced_subject_leaves = {
            tag.rsplit("/", 1)[-1] for tag in remaining_tags_list
        }
        subject_leaves_to_remove = {
            tag.rsplit("/", 1)[-1]
            for tag in to_remove
            if tag.rsplit("/", 1)[-1] not in referenced_subject_leaves
        }

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".xmp", dir=sidecar.parent
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            shutil.copy2(sidecar, temp_path)
            args: list[str | Path] = [
                self.exiftool,
                "-m",
                "-overwrite_original",
            ]
            for tag in to_remove:
                args.extend(
                    [
                        f"-XMP-digiKam:TagsList-={tag}",
                        f"-XMP-lr:HierarchicalSubject-={tag.replace('/', '|')}",
                    ]
                )
            for leaf in sorted(subject_leaves_to_remove, key=str.casefold):
                args.append(f"-XMP-dc:Subject-={leaf}")
            args.append(temp_path)
            run_command(args, timeout=120)

            written = self.read_tag_fields(temp_path)
            tags_list = set(written["TagsList"])
            hierarchical = set(written["HierarchicalSubject"])
            subjects = set(written["Subject"])
            remaining = sorted(tags_list, key=str.casefold)
            for tag in to_remove:
                if tag in tags_list:
                    raise RuntimeError(f"ExifTool did not remove tag: {tag}")
                hierarchical_expected = tag.replace("/", "|")
                if hierarchical_expected in hierarchical:
                    raise RuntimeError(
                        f"ExifTool did not remove HierarchicalSubject: {hierarchical_expected}"
                    )
            for leaf in subject_leaves_to_remove:
                if leaf in subjects:
                    raise RuntimeError(f"ExifTool did not remove Subject: {leaf}")

            os.replace(temp_path, sidecar)
        finally:
            temp_path.unlink(missing_ok=True)

        return MetadataRemovalResult(sidecar, tuple(to_remove), tuple(remaining))
