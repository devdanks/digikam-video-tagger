from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymysql

from .config import DatabaseConfig


@dataclass(frozen=True)
class PersonEmbedding:
    identity_id: int
    name: str
    vector: np.ndarray


@dataclass(frozen=True)
class CatalogFrameFaces:
    frame: Path
    image_id: int | None
    person_tag_paths: tuple[str, ...]
    catalogued: bool = True


class DigiKamFaceGallery:
    """Read-only access to digiKam's SFace training data."""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def _connect(self):
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset="utf8mb4",
            autocommit=True,
            read_timeout=5,
            write_timeout=5,
            connect_timeout=3,
        )

    def load(self) -> list[PersonEmbedding]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT LOWER(TABLE_NAME) FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=%s",
                (self.config.database,),
            )
            tables = {row[0] for row in cursor.fetchall()}
            required = {"facematrices", "identityattributes"}
            if not required.issubset(tables):
                return []

            cursor.execute("SELECT id, attribute, value FROM IdentityAttributes")
            attributes: dict[int, dict[str, str]] = {}
            for identity_id, attribute, value in cursor.fetchall():
                attributes.setdefault(int(identity_id), {})[str(attribute)] = str(
                    value or ""
                )

            cursor.execute("SELECT identity, embedding FROM FaceMatrices")
            gallery: list[PersonEmbedding] = []
            for identity_id, blob in cursor.fetchall():
                identity_id = int(identity_id)
                attrs = attributes.get(identity_id, {})
                name = attrs.get("fullName") or attrs.get("name")
                if not name or not blob:
                    continue
                vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
                if vector.size != 128:
                    continue
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                gallery.append(PersonEmbedding(identity_id, name, vector / norm))
            return gallery


class DigiKamCatalog:
    """Read-only access to confirmed face assignments in the core catalog."""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def _connect(self):
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset="utf8mb4",
            autocommit=True,
            read_timeout=10,
            write_timeout=5,
            connect_timeout=3,
        )

    @staticmethod
    def _filesystem_root(frame: Path, specific_path: str) -> Path:
        root = Path(specific_path.replace("/", os.sep))
        if root.is_absolute():
            return root
        normalized = str(root).lstrip("\\/")
        return Path(f"{frame.drive}{os.sep}{normalized}")

    @staticmethod
    def _relative_album(frame: Path, root: Path) -> str | None:
        frame_parent = str(frame.resolve().parent).rstrip("\\/")
        root_text = str(root.resolve()).rstrip("\\/")
        if frame_parent.casefold() == root_text.casefold():
            return "/"
        prefix = root_text + os.sep
        if not frame_parent.casefold().startswith(prefix.casefold()):
            return None
        relative = frame_parent[len(prefix) :].replace("\\", "/")
        return f"/{relative}"

    @staticmethod
    def _tag_paths(rows: list[tuple[int, int, str]]) -> dict[int, str]:
        tags = {
            int(tag_id): (int(parent_id), str(name)) for tag_id, parent_id, name in rows
        }
        cache: dict[int, str] = {}

        def build(tag_id: int, visiting: set[int] | None = None) -> str:
            if tag_id in cache:
                return cache[tag_id]
            if tag_id not in tags:
                return ""
            visiting = set() if visiting is None else visiting
            if tag_id in visiting:
                raise RuntimeError(f"Cycle in digiKam tag hierarchy at tag {tag_id}")
            visiting.add(tag_id)
            parent_id, name = tags[tag_id]
            parent = build(parent_id, visiting) if parent_id else ""
            visiting.remove(tag_id)
            path = f"{parent}/{name}" if parent else name
            cache[tag_id] = path
            return path

        return {tag_id: build(tag_id) for tag_id in tags}

    def confirmed_faces_for_frames(self, frames: list[Path]) -> list[CatalogFrameFaces]:
        normalized_frames = [frame.resolve() for frame in frames]
        if not normalized_frames:
            return []

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT id, specificPath FROM AlbumRoots WHERE status=0")
            roots = [
                (int(root_id), str(specific_path))
                for root_id, specific_path in cursor.fetchall()
            ]

            locations: dict[Path, tuple[int, str] | None] = {}
            for frame in normalized_frames:
                candidates: list[tuple[int, str, int]] = []
                for root_id, specific_path in roots:
                    root_path = self._filesystem_root(frame, specific_path)
                    album_path = self._relative_album(frame, root_path)
                    if album_path is not None:
                        candidates.append((root_id, album_path, len(str(root_path))))
                if candidates:
                    root_id, album_path, _ = max(candidates, key=lambda item: item[2])
                    locations[frame] = (root_id, album_path)
                else:
                    locations[frame] = None

            image_ids: dict[Path, int] = {}
            grouped: dict[tuple[int, str], list[Path]] = {}
            for frame, location in locations.items():
                if location is not None:
                    grouped.setdefault(location, []).append(frame)

            for (root_id, album_path), grouped_frames in grouped.items():
                placeholders = ",".join(["%s"] * len(grouped_frames))
                cursor.execute(
                    "SELECT i.id, i.name FROM Images i "
                    "JOIN Albums a ON a.id=i.album "
                    f"WHERE a.albumRoot=%s AND a.relativePath=%s AND i.name IN ({placeholders})",
                    (root_id, album_path, *(frame.name for frame in grouped_frames)),
                )
                by_name = {
                    str(name).casefold(): int(image_id)
                    for image_id, name in cursor.fetchall()
                }
                for frame in grouped_frames:
                    image_id = by_name.get(frame.name.casefold())
                    if image_id is not None:
                        image_ids[frame] = image_id

            cursor.execute("SELECT id, pid, name FROM Tags")
            paths = self._tag_paths(
                [
                    (int(tag_id), int(parent_id), str(name))
                    for tag_id, parent_id, name in cursor.fetchall()
                ]
            )

            faces_by_id: dict[int, set[str]] = {
                image_id: set() for image_id in image_ids.values()
            }
            if faces_by_id:
                placeholders = ",".join(["%s"] * len(faces_by_id))
                cursor.execute(
                    "SELECT DISTINCT itp.imageid, itp.tagid "
                    "FROM ImageTagProperties itp "
                    "JOIN TagProperties person ON person.tagid=itp.tagid AND person.property='person' "
                    "WHERE itp.property='tagRegion' "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM TagProperties special "
                    "  WHERE special.tagid=itp.tagid "
                    "  AND special.property IN ('unknownPerson','unconfirmedPerson','ignoredPerson')"
                    ") "
                    f"AND itp.imageid IN ({placeholders})",
                    tuple(faces_by_id),
                )
                for image_id, tag_id in cursor.fetchall():
                    tag_path = paths.get(int(tag_id), "")
                    if tag_path:
                        faces_by_id[int(image_id)].add(tag_path)

        return [
            CatalogFrameFaces(
                frame=frame,
                image_id=image_ids.get(frame),
                person_tag_paths=tuple(
                    sorted(
                        faces_by_id.get(image_ids.get(frame, -1), set()),
                        key=str.casefold,
                    )
                ),
                catalogued=locations[frame] is not None,
            )
            for frame in normalized_frames
        ]

    def face_statistics(self) -> tuple[int, int, int]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM ImageTagProperties WHERE property='tagRegion'),"
                "(SELECT COUNT(DISTINCT tagid) FROM TagProperties WHERE property='person'),"
                "(SELECT COUNT(*) FROM information_schema.TABLES "
                " WHERE TABLE_SCHEMA=%s AND LOWER(TABLE_NAME)='facematrices')",
                (self.config.database,),
            )
            regions, people, has_matrices = cursor.fetchone()
            embeddings = 0
            if has_matrices:
                cursor.execute("SELECT COUNT(*) FROM FaceMatrices")
                embeddings = int(cursor.fetchone()[0])
            return int(regions), int(people), embeddings
