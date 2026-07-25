from __future__ import annotations

PEOPLE_ROOT = "People"
AUTO_TAG_ROOT = "Auto Tags/Video"


def people_tag(name: str) -> str:
    return f"{PEOPLE_ROOT}/{name.strip('/')}"


def object_tag(root: str, label: str) -> str:
    return f"{root.strip('/')}/Objects/{label}"


def contains_faces_tag(root: str) -> str:
    return f"{root.strip('/')}/Contains Faces"
