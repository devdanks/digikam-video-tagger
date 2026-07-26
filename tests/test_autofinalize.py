from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

from digikam_video_tagger.autofinalize import (
    AutoFinalizeOptions,
    AutoFinalizeService,
)
from digikam_video_tagger.clustering import FaceClusterStore
from digikam_video_tagger.digikam_db import PersonEmbedding
from digikam_video_tagger.ffmpeg import VideoInfo
from digikam_video_tagger.jobs import (
    CompletedVideo,
    PreparedJob,
    ProxyFrame,
    VideoFaceJob,
    job_id_for_video,
    mark_job_completed,
    rewrite_completed_entry,
)
from digikam_video_tagger.models import FaceDetection


def unit(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def known_detection(name: str, confidence: float) -> FaceDetection:
    return FaceDetection(name=name, confidence=confidence, embedding=unit(0))


def unknown_detection(embedding: np.ndarray, confidence: float = 0.91) -> FaceDetection:
    return FaceDetection(name=None, confidence=confidence, embedding=embedding)


@contextmanager
def null_lock(_staging_dir: Path) -> Iterator[None]:
    yield


class FakeVideoFaceJob(VideoFaceJob):
    _remove_callback: object = None

    def remove_generated_files(self) -> list[Path]:
        if self._remove_callback is not None:
            self._remove_callback()
        return super().remove_generated_files()


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
    def __init__(
        self,
        frame_detections: list[list[FaceDetection]],
        gallery: list[PersonEmbedding] | None = None,
    ) -> None:
        self._frame_detections = frame_detections
        self._index = 0
        self.gallery = gallery or []
        self.recognition_distance = 0.50

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
            lock=null_lock,
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


def test_resolution_replaces_owned_placeholders_on_apply(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    store = FaceClusterStore.empty(
        model_fingerprint="a" * 64,
        distance_threshold=0.20,
        unknown_root="People/Unknown",
    )
    session = store.begin_session()
    token = session.assign(unit(0))
    session.commit({token})
    store.save(tmp_path / "clusters.json")

    videos: list[Path] = []
    for name in ("a.mp4", "b.mp4"):
        video = tmp_path / name
        video.write_bytes(b"video")
        sidecar = Path(f"{video}.xmp")
        sidecar.write_text("xmp", encoding="utf-8")
        stat = video.stat()
        rewrite_completed_entry(
            staging,
            CompletedVideo(
                job_id=job_id_for_video(video),
                source_video=str(video.resolve()),
                source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                applied_at="2026-01-01T00:00:00+00:00",
                people=("People/Unknown/Person_001",),
                sidecar=str(sidecar),
                workflow="autofinalize",
                managed_placeholders=("People/Unknown/Person_001",),
                managed_placeholder_root="People/Unknown",
                cluster_store_id=store.store_id,
            ),
        )
        videos.append(video)

    tagger = FakeTagger([[]], gallery=[PersonEmbedding(1, "Mom", unit(0))])
    writer = FakeWriter()
    sampler = FakeSampler([])
    service = AutoFinalizeService(
        staging_dir=staging,
        sampler=sampler,
        face_tagger=tagger,
        cluster_store=store,
        cluster_store_path=tmp_path / "clusters.json",
        sidecar_writer=writer,
        options=AutoFinalizeOptions(
            sample_seconds=1.0,
            max_frames=10,
            max_dimension=1280,
            min_person_hits=1,
            min_frame_ratio=0.01,
            resolution_min_observations=1,
            resolution_margin=0.001,
        ),
        lock=null_lock,
    )

    results, summary = service.run(videos, apply=False, reprocess_completed=False)
    assert summary.resolved_people == 2
    assert all(len(r.proposed_replacements) == 1 for r in results)
    assert writer.writes == []
    assert writer.removals == []

    results, summary = service.run(videos, apply=True, reprocess_completed=False)
    assert summary.resolved_people == 1
    assert len(writer.writes) == 2
    assert len(writer.removals) == 2
    assert all("People/Mom" in tags for _video, tags in writer.writes)
    assert all("People/Unknown/Person_001" in tags for _video, tags in writer.removals)

    from digikam_video_tagger import jobs

    completed = jobs.load_completed_videos(staging)
    for video in videos:
        entry = completed[job_id_for_video(video)]
        assert "People/Mom" in entry.people
        assert "People/Unknown/Person_001" not in entry.managed_placeholders

    assert store.clusters["Person_001"].resolved_name == "Mom"


def _make_fake_job(
    video: Path,
    staging_dir: Path,
    frame_count: int,
    callback: object,
) -> VideoFaceJob:
    job_dir = staging_dir / f"job_{job_id_for_video(video)}"
    job_dir.mkdir(parents=True, exist_ok=True)
    frame_names: list[str] = []
    for index in range(frame_count):
        frame_path = job_dir / f"frame_{index + 1:06d}.jpg"
        cv2.imwrite(str(frame_path), np.zeros((16, 16, 3), dtype=np.uint8))
        frame_names.append(frame_path.name)
    stat = video.stat()
    job = FakeVideoFaceJob(
        schema_version=1,
        job_id=job_id_for_video(video),
        source_video=str(video),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        created_at="2026-01-01T00:00:00+00:00",
        sample_seconds=1.0,
        video_duration_seconds=1.0,
        video_codec="h264",
        frames=tuple(ProxyFrame(name, 0.0) for name in frame_names),
        job_dir=job_dir,
    )
    object.__setattr__(job, "_remove_callback", callback)
    return job


def test_apply_ordering_is_cluster_save_sidecar_ledger_cleanup(
    make_service, tmp_path: Path
) -> None:
    events: list[str] = []

    def fake_prepare_job(
        video: Path, staging_dir: Path, sampler: object, **kwargs
    ) -> PreparedJob:
        job = _make_fake_job(
            video,
            staging_dir,
            frame_count=1,
            callback=lambda: events.append("proxy-cleanup"),
        )
        return PreparedJob(job, True, VideoInfo(video, 1.0, 16, 16, "h264"))

    service, video, store, writer, _tagger = make_service(
        frame_detections=[[unknown_detection(unit(0))]]
    )
    service.prepare_job = fake_prepare_job

    original_save = store.save
    store.save = lambda path: events.append("cluster-save") or original_save(path)
    original_write = writer.write_tags
    writer.write_tags = lambda v, tags: (
        events.append("sidecar-write"),
        original_write(v, tags),
    )[1]

    def marked_completed(*args, **kwargs) -> CompletedVideo:
        events.append("ledger-write")
        return mark_job_completed(*args, **kwargs)

    service.mark_completed = marked_completed

    service.run([video], apply=True, reprocess_completed=False)
    assert events == [
        "cluster-save",
        "sidecar-write",
        "ledger-write",
        "proxy-cleanup",
    ]


def test_no_tags_apply_still_marks_completion_and_cleans(
    make_service, tmp_path: Path
) -> None:
    events: list[str] = []

    def fake_prepare_job(
        video: Path, staging_dir: Path, sampler: object, **kwargs
    ) -> PreparedJob:
        job = _make_fake_job(
            video,
            staging_dir,
            frame_count=1,
            callback=lambda: events.append("proxy-cleanup"),
        )
        return PreparedJob(job, True, VideoInfo(video, 1.0, 16, 16, "h264"))

    service, video, store, writer, _tagger = make_service(frame_detections=[[]])
    service.prepare_job = fake_prepare_job

    results, _summary = service.run([video], apply=True, reprocess_completed=False)
    assert results[0].applied is True
    assert results[0].completed is True
    assert "proxy-cleanup" in events
    assert writer.writes == []
    assert store.saved is False


def test_dry_run_extracts_frames_at_most_once(make_service) -> None:
    service, video, _store, _writer, _tagger = make_service(frame_detections=[[]])
    original = service.sampler.extract_frames
    calls = {"count": 0}

    def counting_extract(video, **kwargs):
        calls["count"] += 1
        return original(video, **kwargs)

    service.sampler.extract_frames = counting_extract
    service.run([video], apply=False, reprocess_completed=False)
    assert calls["count"] == 1


def test_resolution_failure_surfaces_and_leaves_cluster_unresolved(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    store = FaceClusterStore.empty(
        model_fingerprint="a" * 64,
        distance_threshold=0.20,
        unknown_root="People/Unknown",
    )
    session = store.begin_session()
    token = session.assign(unit(0))
    session.commit({token})
    store.save(tmp_path / "clusters.json")

    videos: list[Path] = []
    for name in ("a.mp4", "b.mp4"):
        video = tmp_path / name
        video.write_bytes(b"video")
        sidecar = Path(f"{video}.xmp")
        sidecar.write_text("xmp", encoding="utf-8")
        stat = video.stat()
        rewrite_completed_entry(
            staging,
            CompletedVideo(
                job_id=job_id_for_video(video),
                source_video=str(video.resolve()),
                source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                applied_at="2026-01-01T00:00:00+00:00",
                people=("People/Unknown/Person_001",),
                sidecar=str(sidecar),
                workflow="autofinalize",
                managed_placeholders=("People/Unknown/Person_001",),
                managed_placeholder_root="People/Unknown",
                cluster_store_id=store.store_id,
            ),
        )
        videos.append(video)

    class FailingRemoveWriter:
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
            raise RuntimeError("exiftool remove failed")

    tagger = FakeTagger([[]], gallery=[PersonEmbedding(1, "Mom", unit(0))])
    writer = FailingRemoveWriter()
    service = AutoFinalizeService(
        staging_dir=staging,
        sampler=FakeSampler([]),
        face_tagger=tagger,
        cluster_store=store,
        cluster_store_path=tmp_path / "clusters.json",
        sidecar_writer=writer,
        options=AutoFinalizeOptions(
            sample_seconds=1.0,
            max_frames=10,
            max_dimension=1280,
            min_person_hits=1,
            min_frame_ratio=0.01,
            resolution_min_observations=1,
            resolution_margin=0.001,
        ),
        lock=null_lock,
    )

    with pytest.raises(RuntimeError, match="Placeholder resolution failed"):
        service.run(videos, apply=True, reprocess_completed=False)

    assert store.clusters["Person_001"].resolved_name is None
    assert len(writer.writes) == 2
    assert len(writer.removals) == 2
