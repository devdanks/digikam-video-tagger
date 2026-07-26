from __future__ import annotations

# ruff: noqa: TRY004
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .digikam_db import PersonEmbedding
from .tags import validate_tag_path

CLUSTER_STORE_VERSION = 1
CLUSTER_STORE_NAME = ".digikam-video-face-clusters.json"
EMBEDDING_DIMENSION = 128
_ID_PATTERN = re.compile(r"^Person_(\d{3,})$")
_NORMALIZATION_TOLERANCE = 1e-5


@dataclass
class FaceCluster:
    count: int
    centroid: np.ndarray
    resolved_name: str | None = None


@dataclass(frozen=True)
class SessionObservation:
    token: str
    embedding: np.ndarray
    persistent_id: str | None


@dataclass
class ClusterSession:
    _store: FaceClusterStore
    _persistent_centroids: dict[str, np.ndarray] = field(default_factory=dict)
    _session_clusters: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict)
    _token_embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _next_session_id: int = 1

    def __post_init__(self) -> None:
        for cluster_id, cluster in self._store.clusters.items():
            if cluster.resolved_name is None:
                self._persistent_centroids[cluster_id] = cluster.centroid.copy()

    def assign(self, embedding: np.ndarray) -> str:
        if embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"embedding must have dimension {EMBEDDING_DIMENSION}")

        best_persistent = self._nearest_persistent(embedding)
        if best_persistent is not None:
            token = f"persistent:{best_persistent}"
            self._token_embeddings.setdefault(token, []).append(embedding)
            return token

        best_session = self._nearest_session(embedding)
        if best_session is not None:
            token = f"session:{best_session}"
            centroid, count = self._session_clusters[best_session]
            new_centroid = (centroid * count + embedding) / (count + 1)
            new_centroid /= np.linalg.norm(new_centroid)
            self._session_clusters[best_session] = (new_centroid, count + 1)
            self._token_embeddings.setdefault(token, []).append(embedding)
            return token

        local_id = self._next_session_id
        self._next_session_id += 1
        normalized = embedding / np.linalg.norm(embedding)
        self._session_clusters[str(local_id)] = (normalized, 1)
        token = f"session:{local_id}"
        self._token_embeddings.setdefault(token, []).append(embedding)
        return token

    def _nearest_persistent(self, embedding: np.ndarray) -> str | None:
        return self._nearest(
            embedding,
            (
                (cluster_id, centroid)
                for cluster_id, centroid in self._persistent_centroids.items()
            ),
        )

    def _nearest_session(self, embedding: np.ndarray) -> str | None:
        return self._nearest(
            embedding,
            (
                (cluster_id, centroid)
                for cluster_id, (centroid, _count) in self._session_clusters.items()
            ),
        )

    def _nearest(
        self,
        embedding: np.ndarray,
        candidates: Iterator[tuple[str, np.ndarray]],
    ) -> str | None:
        threshold = self._store.distance_threshold
        best_id: str | None = None
        best_distance = float("inf")
        for cluster_id, centroid in candidates:
            distance = 1.0 - float(np.dot(embedding, centroid))
            l2_distance = float(np.linalg.norm(embedding - centroid))
            if distance < threshold and l2_distance < 1.05 and distance < best_distance:
                best_id = cluster_id
                best_distance = distance
        return best_id

    def commit(self, accepted_tokens: set[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for token in accepted_tokens:
            embeddings = self._token_embeddings.get(token)
            if not embeddings:
                continue
            centroid = np.mean(np.stack(embeddings), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm <= 0:
                continue
            centroid = centroid / norm

            if token.startswith("persistent:"):
                cluster_id = token.split(":", 1)[1]
                cluster = self._store.clusters[cluster_id]
                combined = (
                    cluster.centroid * cluster.count + centroid * len(embeddings)
                ) / (cluster.count + len(embeddings))
                combined /= np.linalg.norm(combined)
                cluster.centroid = combined
                cluster.count += len(embeddings)
                result[token] = f"{self._store.unknown_root}/{cluster_id}"
            elif token.startswith("session:"):
                cluster_id = self._store._allocate_id()
                self._store.clusters[cluster_id] = FaceCluster(
                    count=len(embeddings), centroid=centroid
                )
                result[token] = f"{self._store.unknown_root}/{cluster_id}"
        return result


class FaceClusterStore:
    def __init__(
        self,
        *,
        store_id: str,
        model_fingerprint: str,
        embedding_dimension: int,
        distance_threshold: float,
        unknown_root: str,
        next_id: int,
        clusters: dict[str, FaceCluster],
    ) -> None:
        self.store_id = store_id
        self.model_fingerprint = model_fingerprint
        self.embedding_dimension = embedding_dimension
        self.distance_threshold = distance_threshold
        self.unknown_root = unknown_root
        self.next_id = next_id
        self.clusters = clusters

    @classmethod
    def empty(
        cls,
        *,
        model_fingerprint: str,
        distance_threshold: float,
        unknown_root: str,
    ) -> FaceClusterStore:
        validate_tag_path(unknown_root)
        return cls(
            store_id=uuid.uuid4().hex,
            model_fingerprint=model_fingerprint,
            embedding_dimension=EMBEDDING_DIMENSION,
            distance_threshold=distance_threshold,
            unknown_root=unknown_root,
            next_id=1,
            clusters={},
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        model_fingerprint: str,
        distance_threshold: float,
        unknown_root: str,
    ) -> FaceClusterStore:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("cluster store root must be an object")

        version = data.get("schema_version")
        if version != CLUSTER_STORE_VERSION:
            raise ValueError(f"cluster store schema version {version} is not supported")

        loaded_fingerprint = data.get("model_fingerprint")
        if loaded_fingerprint != model_fingerprint:
            raise ValueError(
                f"SFace model fingerprint mismatch: store used {loaded_fingerprint!r}, "
                f"current model is {model_fingerprint!r}"
            )

        loaded_threshold = data.get("distance_threshold")
        if not isinstance(loaded_threshold, (int, float)) or isinstance(
            loaded_threshold, bool
        ):
            raise ValueError("distance_threshold must be a number")
        if loaded_threshold != distance_threshold:
            raise ValueError(
                f"distance threshold mismatch: store has {loaded_threshold}, "
                f"requested {distance_threshold}"
            )

        loaded_root = data.get("unknown_root")
        if not isinstance(loaded_root, str):
            raise ValueError("unknown_root must be a string")
        if loaded_root != unknown_root:
            raise ValueError(
                f"unknown root mismatch: store has {loaded_root!r}, "
                f"requested {unknown_root!r}"
            )
        validate_tag_path(loaded_root)

        embedding_dimension = data.get("embedding_dimension")
        if not isinstance(embedding_dimension, int) or isinstance(
            embedding_dimension, bool
        ):
            raise ValueError("embedding_dimension must be an integer")
        if embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"embedding dimension mismatch: store has {embedding_dimension}, "
                f"expected {EMBEDDING_DIMENSION}"
            )

        next_id = data.get("next_id")
        if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 1:
            raise ValueError("next_id must be a positive integer")

        store_id = data.get("store_id")
        if not isinstance(store_id, str) or not store_id:
            raise ValueError("store_id must be a non-empty string")

        raw_clusters = data.get("clusters")
        if not isinstance(raw_clusters, dict):
            raise ValueError("clusters must be an object")

        clusters: dict[str, FaceCluster] = {}
        seen_numbers: set[int] = set()
        for cluster_id, raw in raw_clusters.items():
            if not isinstance(raw, dict):
                raise ValueError(f"cluster {cluster_id} must be an object")
            match = _ID_PATTERN.match(cluster_id)
            if match is None:
                raise ValueError(f"invalid cluster id: {cluster_id}")
            number = int(match.group(1))
            if number in seen_numbers:
                raise ValueError(f"duplicate cluster id: {cluster_id}")
            seen_numbers.add(number)

            count = raw.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"cluster {cluster_id} count must be a positive integer"
                )

            centroid = np.asarray(raw.get("centroid"), dtype=np.float32)
            if centroid.shape != (EMBEDDING_DIMENSION,):
                raise ValueError(
                    f"cluster {cluster_id} centroid must have dimension {EMBEDDING_DIMENSION}"
                )
            if not np.isfinite(centroid).all():
                raise ValueError(f"cluster {cluster_id} centroid must be finite")
            norm = float(np.linalg.norm(centroid))
            if abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
                raise ValueError(f"cluster {cluster_id} centroid must be normalized")
            centroid = centroid / norm

            resolved_name = raw.get("resolved_name")
            if resolved_name is not None and not isinstance(resolved_name, str):
                raise ValueError(
                    f"cluster {cluster_id} resolved_name must be a string or null"
                )

            clusters[cluster_id] = FaceCluster(
                count=count, centroid=centroid, resolved_name=resolved_name
            )

        if seen_numbers and next_id <= max(seen_numbers):
            raise ValueError(
                f"next_id {next_id} must be greater than all allocated cluster ids"
            )

        return cls(
            store_id=store_id,
            model_fingerprint=model_fingerprint,
            embedding_dimension=embedding_dimension,
            distance_threshold=distance_threshold,
            unknown_root=unknown_root,
            next_id=next_id,
            clusters=clusters,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": CLUSTER_STORE_VERSION,
            "store_id": self.store_id,
            "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": self.embedding_dimension,
            "distance_threshold": self.distance_threshold,
            "unknown_root": self.unknown_root,
            "next_id": self.next_id,
            "clusters": {
                cluster_id: {
                    "count": cluster.count,
                    "centroid": cluster.centroid.tolist(),
                    "resolved_name": cluster.resolved_name,
                }
                for cluster_id, cluster in self.clusters.items()
            },
        }
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="clusters-", dir=str(path.parent)
        )
        temp_file = Path(temp_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def begin_session(self) -> ClusterSession:
        return ClusterSession(self)

    def _allocate_id(self) -> str:
        cluster_id = f"Person_{self.next_id:03d}"
        self.next_id += 1
        return cluster_id

    def resolution_candidates(
        self,
        gallery: list[PersonEmbedding],
        *,
        recognition_distance: float,
        min_observations: int,
        min_margin: float,
    ) -> dict[str, str]:
        candidates: dict[str, str] = {}
        for cluster_id, cluster in self.clusters.items():
            if cluster.resolved_name is not None:
                continue
            if cluster.count < min_observations:
                continue

            best_name: str | None = None
            best_distance = float("inf")
            second_best_distance = float("inf")
            for sample in gallery:
                distance = 1.0 - float(np.dot(cluster.centroid, sample.vector))
                l2_distance = float(np.linalg.norm(cluster.centroid - sample.vector))
                if distance >= recognition_distance or l2_distance >= 1.05:
                    continue
                if distance < best_distance:
                    second_best_distance = best_distance
                    best_distance = distance
                    best_name = sample.name
                elif distance < second_best_distance:
                    second_best_distance = distance

            if (
                best_name is not None
                and second_best_distance - best_distance >= min_margin
            ):
                candidates[f"{self.unknown_root}/{cluster_id}"] = best_name
        return candidates
