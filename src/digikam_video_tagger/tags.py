from __future__ import annotations

PEOPLE_ROOT = "People"
AUTO_TAG_ROOT = "Auto Tags/Video"


def _tag_segment(value: str, *, name: str) -> str:
    segment = value.strip()
    if not segment or "/" in segment or "|" in segment:
        raise ValueError(f"{name} must be a non-empty tag segment without '/' or '|'")
    return segment


def validate_tag_path(value: str) -> str:
    segments = value.split("/")
    if not segments:
        raise ValueError("tag path must not be empty")
    return "/".join(
        _tag_segment(segment, name="tag path segment") for segment in segments
    )


def people_tag(name: str) -> str:
    return f"{PEOPLE_ROOT}/{_tag_segment(name, name='name')}"


def object_tag(root: str, label: str) -> str:
    return f"{root.strip('/')}/Objects/{_tag_segment(label, name='label')}"


def contains_faces_tag(root: str) -> str:
    return f"{root.strip('/')}/Contains Faces"
