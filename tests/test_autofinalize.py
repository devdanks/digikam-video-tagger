from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from digikam_video_tagger.autofinalize import (
    AutoFinalizeOptions,
    AutoFinalizeService,
)
from digikam_video_tagger.clustering import FaceClusterStore
from digikam_video_tagger.ffmpeg import VideoInfo
from digikam_video_tagger.models import FaceDetection


def unit(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def known_detection(name: str, confidence: float) -> FaceDetection:
    return FaceDetection(name=name, confidence=confidence, embedding=unit(0))


def unknown_detection(embedding: np.ndarray, confidence: float = 0.91) -> FaceDetection:
    return FaceDetection(name=None, confidence=confidence, embedding=embedding)


class FakeSampler:
    def __init__(self, frames: list[Path]) -> None:
        self._frames = frames

    def extract_frames(
        self,
        video: Path,
        *,
        sample_seconds: float,
        max_frames: int,
        max_dimension: int,
    ) -> tuple[VideoInfo, list[Path], tempfile.TemporaryDirectory[str] | None]:
        return VideoInfo(video, 1.0, 16, 16, "h264"), self._frames, None

    def probe(self, video: Path) -> VideoInfo:
        return VideoInfo(video, 1.0, 16, 16, "h264")


class FakeTagger:
    def __init__(self, frame_detections: list[list[FaceDetection]]) -> None:
        self._frame_detections = frame_detections
        self._index = 0

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        if self._index >= len(self._frame_detections):
            return []
        detections = self._frame_detections[self._index]
        self._index += 1
        return detections


class FakeWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, list[str]]] = []
        self.removals: list[tuple[Path, list[str]]] = []

    def sidecar_path(self, video: Path) -> Path:
        return Path(f"{video}.xmp")

    def write_tags(self, video: Path, tags: list[str]) -> object:
        self.writes.append((video, tags))
        return type("Result", (), {"sidecar": self.sidecar_path(video)})()

    def remove_tags(self, video: Path, tags: list[str]) -> object:
        self.removals.append((video, tags))
        return type("Result", (), {"sidecar": self.sidecar_path(video)})()


class RecordingEvidenceAccumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.labels: list[dict[str, float]] = []

    def add_frame(self, labels: dict[str, float]) -> None:
        self.frames += 1
        self.labels.append(labels)

    def accepted(
        self, *, min_hits: int, min_frame_ratio: float, limit: int | None = None
    ) -> list:
        from digikam_video_tagger.evidence import EvidenceAccumulator

        real = EvidenceAccumulator()
        for labels in self.labels:
            real.add_frame(labels)
        return real.accepted(min_hits=min_hits, min_frame_ratio=min_frame_ratio)


class FakeClusterSession:
    def __init__(self, store: FaceClusterStore) -> None:
        self._store = store
        self._next = 1
        self._centroids: dict[str, np.ndarray] = {}
        self._token_embeddings: dict[str, list[np.ndarray]] = {}
        self.committed: set[str] = set()

    def assign(self, embedding: np.ndarray) -> str:
        for token, centroid in self._centroids.items():
            if 1.0 - float(np.dot(embedding, centroid)) < 0.20:
                self._token_embeddings.setdefault(token, []).append(embedding)
                return token
        token = f"session:{self._next}"
        self._next += 1
        self._centroids[token] = embedding / np.linalg.norm(embedding)
        self._token_embeddings.setdefault(token, []).append(embedding)
        return token

    def commit(self, accepted_tokens: set[str]) -> dict[str, str]:
        self.committed.update(accepted_tokens)
        result: dict[str, str] = {}
        for token in accepted_tokens:
            result[token] = f"{self._store.unknown_root}/Person_001"
        return result


class FakeClusterStore:
    def __init__(self) -> None:
        self.unknown_root = "People/Unknown"
        self.store_id = "fake-store"
        self._session: FakeClusterSession | None = None
        self.saved = False

    def begin_session(self) -> FakeClusterSession:
        self._session = FakeClusterSession(self)
        return self._session

    def save(self, path: Path) -> None:
        self.saved = True


@pytest.fixture
def make_service(tmp_path: Path):
    def _make(
        frame_detections: list[list[FaceDetection]],
        *,
        options: AutoFinalizeOptions | None = None,
        cluster_store: FakeClusterStore | None = None,
    ) -> tuple[AutoFinalizeService, Path, FakeClusterStore, FakeWriter, FakeTagger]:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_paths: list[Path] = []
        for index in range(len(frame_detections)):
            frame_path = frames_dir / f"frame_{index + 1:06d}.jpg"
            cv2.imwrite(str(frame_path), np.zeros((16, 16, 3), dtype=np.uint8))
            frame_paths.append(frame_path)

        sampler = FakeSampler(frame_paths)
        tagger = FakeTagger(frame_detections)
        store = cluster_store or FakeClusterStore()
        writer = FakeWriter()
        service = AutoFinalizeService(
            staging_dir=tmp_path / "staging",
            sampler=sampler,
            face_tagger=tagger,
            cluster_store=store,
            cluster_store_path=tmp_path / "clusters.json",
            sidecar_writer=writer,
            options=options
            or AutoFinalizeOptions(
                sample_seconds=1.0,
                max_frames=10,
                max_dimension=1280,
                min_person_hits=1,
                min_frame_ratio=0.01,
            ),
        )
        return service, video, store, writer, tagger

    return _make


def test_analysis_adds_evidence_once_per_proxy_frame(make_service) -> None:
    service, video, _store, _writer, _tagger = make_service(
        frame_detections=[
            [
                known_detection("Mom", 0.95),
                unknown_detection(unit(0)),
                unknown_detection(unit(0)),
            ]
        ]
    )
    results, summary = service.run([video], apply=False, reprocess_completed=False)
    assert results[0].frame_count == 1
    assert results[0].known_people == ("People/Mom",)
    assert len(results[0].placeholder_people) == 1
    assert summary.failed == 0


def test_readable_no_face_frame_remains_in_evidence_denominator(make_service) -> None:
    service, video, _store, _writer, _tagger = make_service(
        frame_detections=[[], [known_detection("Mom", 0.95)]]
    )
    results, _summary = service.run([video], apply=False, reprocess_completed=False)
    assert results[0].frame_count == 2
    assert results[0].face_frames == 1
    assert results[0].known_people == ("People/Mom",)


def test_unreadable_frame_remains_in_evidence_denominator(
    make_service, tmp_path: Path
) -> None:
    service, video, _store, _writer, _tagger = make_service(
        frame_detections=[[known_detection("Mom", 0.95)]]
    )
    # Replace the generated frame with a file cv2 cannot decode.
    service.sampler._frames[0].write_bytes(b"not an image")
    results, _summary = service.run([video], apply=False, reprocess_completed=False)
    assert results[0].frame_count == 1
    assert results[0].unreadable_frames == 1
    assert results[0].known_people == ()


def test_no_face_video_is_a_successful_result(make_service) -> None:
    service, video, _store, _writer, _tagger = make_service(frame_detections=[[]])
    results, summary = service.run([video], apply=False, reprocess_completed=False)
    assert results[0].frame_count == 1
    assert results[0].known_people == ()
    assert results[0].placeholder_people == ()
    assert summary.failed == 0


def test_apply_saves_cluster_store_writes_sidecar_and_cleans(
    make_service, tmp_path: Path
) -> None:
    service, video, store, writer, _tagger = make_service(
        frame_detections=[[unknown_detection(unit(0))]]
    )
    results, _summary = service.run([video], apply=True, reprocess_completed=False)
    assert results[0].applied is True
    assert results[0].completed is True
    assert store.saved is True
    assert writer.writes == [(video, ["People/Unknown/Person_001"])]
