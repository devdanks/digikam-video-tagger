# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .ffmpeg import FFmpegSampler, VideoInfo
from .tags import validate_tag_path

MANIFEST_NAME = ".digikam-video-face-job.json"
MANIFEST_VERSION = 1
COMPLETED_NAME = ".digikam-video-face-completed.json"
COMPLETED_VERSION = 2


@dataclass(frozen=True)
class ProxyFrame:
    filename: str
    timestamp_seconds: float


@dataclass(frozen=True)
class VideoFaceJob:
    schema_version: int
    job_id: str
    source_video: str
    source_size: int
    source_mtime_ns: int
    created_at: str
    sample_seconds: float
    video_duration_seconds: float
    video_codec: str
    frames: tuple[ProxyFrame, ...]
    job_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.job_dir / MANIFEST_NAME

    @property
    def source_path(self) -> Path:
        return Path(self.source_video)

    @property
    def frame_paths(self) -> list[Path]:
        return [self.job_dir / frame.filename for frame in self.frames]

    def source_is_unchanged(self) -> bool:
        try:
            stat = self.source_path.stat()
        except OSError:
            return False
        return (
            stat.st_size == self.source_size
            and stat.st_mtime_ns == self.source_mtime_ns
        )

    def remove_generated_files(self) -> list[Path]:
        """Remove only files named by this manifest and their sidecars."""
        job_root = self.job_dir.resolve()
        removed: list[Path] = []
        for frame_path in self.frame_paths:
            resolved = frame_path.resolve()
            if resolved.parent != job_root:
                raise RuntimeError(
                    f"Unsafe proxy path outside job directory: {resolved}"
                )
            for generated in (resolved, Path(f"{resolved}.xmp")):
                if generated.exists():
                    generated.unlink()
                    removed.append(generated)

        if self.manifest_path.exists():
            self.manifest_path.unlink()
            removed.append(self.manifest_path)
        try:
            self.job_dir.rmdir()
        except OSError:
            pass
        return removed

    @classmethod
    def load(cls, manifest_path: Path) -> VideoFaceJob:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported proxy manifest version in {manifest_path}")
        frames = tuple(ProxyFrame(**frame) for frame in payload.pop("frames"))
        job_root = manifest_path.parent.resolve()
        for frame in frames:
            frame_path = Path(frame.filename)
            resolved = (job_root / frame_path).resolve()
            if (
                frame_path.is_absolute()
                or len(frame_path.parts) != 1
                or resolved.parent != job_root
            ):
                raise ValueError(f"Unsafe proxy frame filename: {frame.filename}")
        return cls(**payload, frames=frames, job_dir=manifest_path.parent)


@dataclass(frozen=True)
class PreparedJob:
    job: VideoFaceJob
    created: bool
    info: VideoInfo


@dataclass(frozen=True)
class CompletedVideo:
    job_id: str
    source_video: str
    source_size: int
    source_mtime_ns: int
    applied_at: str
    people: tuple[str, ...]
    sidecar: str | None
    workflow: str = "confirmed"
    managed_placeholders: tuple[str, ...] = ()
    managed_placeholder_root: str | None = None
    cluster_store_id: str | None = None

    @property
    def source_path(self) -> Path:
        return Path(self.source_video)

    def source_is_unchanged(self) -> bool:
        try:
            stat = self.source_path.stat()
        except OSError:
            return False
        return (
            stat.st_size == self.source_size
            and stat.st_mtime_ns == self.source_mtime_ns
        )


def job_id_for_video(video: Path) -> str:
    canonical = str(video.resolve()).casefold().encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def job_folder_name(video: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video.stem).strip("._-") or "video"
    return f"{stem[:80]}__{job_id_for_video(video)}"


def load_completed_videos(staging_dir: Path) -> dict[str, CompletedVideo]:
    ledger = staging_dir / COMPLETED_NAME
    if not ledger.is_file():
        return {}
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in (1, COMPLETED_VERSION):
        raise ValueError(f"Unsupported completion ledger version in {ledger}")
    result: dict[str, CompletedVideo] = {}
    for job_id, entry in payload.get("videos", {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"completion entry for {job_id} must be an object")

        people = tuple(entry.get("people", []))
        for tag in people:
            validate_tag_path(tag)

        if schema_version == 1:
            managed_placeholders: tuple[str, ...] = ()
            managed_placeholder_root: str | None = None
            workflow = "confirmed"
            cluster_store_id: str | None = None
        else:
            workflow = entry.get("workflow", "confirmed")
            if not isinstance(workflow, str):
                raise ValueError(f"workflow for {job_id} must be a string")
            managed_placeholders = tuple(entry.get("managed_placeholders", []))
            for tag in managed_placeholders:
                validate_tag_path(tag)
            managed_placeholder_root = entry.get("managed_placeholder_root")
            if managed_placeholder_root is not None:
                validate_tag_path(managed_placeholder_root)
                root_prefix = f"{managed_placeholder_root}/"
                for tag in managed_placeholders:
                    if not tag.startswith(root_prefix):
                        raise ValueError(
                            f"managed placeholder {tag} must be a descendant of "
                            f"{managed_placeholder_root}"
                        )
            cluster_store_id = entry.get("cluster_store_id")
            if cluster_store_id is not None and not isinstance(cluster_store_id, str):
                raise ValueError(f"cluster_store_id for {job_id} must be a string")

        result[str(job_id)] = CompletedVideo(
            job_id=str(job_id),
            source_video=entry["source_video"],
            source_size=entry["source_size"],
            source_mtime_ns=entry["source_mtime_ns"],
            applied_at=entry["applied_at"],
            people=people,
            sidecar=entry.get("sidecar"),
            workflow=workflow,
            managed_placeholders=managed_placeholders,
            managed_placeholder_root=managed_placeholder_root,
            cluster_store_id=cluster_store_id,
        )
    return result


def completed_video_for_source(staging_dir: Path, video: Path) -> CompletedVideo | None:
    video = video.resolve()
    entry = load_completed_videos(staging_dir).get(job_id_for_video(video))
    if (
        entry is None
        or entry.source_path.resolve() != video
        or not entry.source_is_unchanged()
    ):
        return None
    return entry


def mark_job_completed(
    job: VideoFaceJob,
    people: list[str],
    sidecar: Path | None,
    *,
    workflow: str = "confirmed",
    managed_placeholders: tuple[str, ...] = (),
    managed_placeholder_root: str | None = None,
    cluster_store_id: str | None = None,
) -> CompletedVideo:
    staging_dir = job.job_dir.parent
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger = staging_dir / COMPLETED_NAME
    existing = load_completed_videos(staging_dir)

    for tag in people:
        validate_tag_path(tag)
    for tag in managed_placeholders:
        validate_tag_path(tag)
    if managed_placeholder_root is not None:
        validate_tag_path(managed_placeholder_root)
        root_prefix = f"{managed_placeholder_root}/"
        for tag in managed_placeholders:
            if not tag.startswith(root_prefix):
                raise ValueError(
                    f"managed placeholder {tag} must be a descendant of "
                    f"{managed_placeholder_root}"
                )

    entry = CompletedVideo(
        job_id=job.job_id,
        source_video=job.source_video,
        source_size=job.source_size,
        source_mtime_ns=job.source_mtime_ns,
        applied_at=datetime.now(UTC).isoformat(),
        people=tuple(sorted(set(people), key=str.casefold)),
        sidecar=str(sidecar) if sidecar is not None else None,
        workflow=workflow,
        managed_placeholders=managed_placeholders,
        managed_placeholder_root=managed_placeholder_root,
        cluster_store_id=cluster_store_id,
    )
    existing[job.job_id] = entry
    payload = {
        "schema_version": COMPLETED_VERSION,
        "videos": {
            job_id: {
                key: value for key, value in asdict(item).items() if key != "job_id"
            }
            for job_id, item in sorted(existing.items())
        },
    }
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".completed-", suffix=".json", dir=staging_dir
    )
    os.close(descriptor)
    temp_ledger = Path(temp_name)
    try:
        temp_ledger.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temp_ledger, ledger)
    finally:
        temp_ledger.unlink(missing_ok=True)
    return entry


@contextmanager
def staging_apply_lock(staging_dir: Path) -> Iterator[None]:
    if sys.platform != "win32":
        raise RuntimeError(
            "Apply locking is only implemented on Windows in this release"
        )
    import msvcrt

    staging_dir.mkdir(parents=True, exist_ok=True)
    lock_path = staging_dir / ".digikam-video-tagger.lock"
    handle = lock_path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(
                f"Another apply operation is using staging directory: {staging_dir}"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def prepare_job(
    video: Path,
    staging_dir: Path,
    sampler: FFmpegSampler,
    *,
    sample_seconds: float,
    max_frames: int,
    max_dimension: int,
) -> PreparedJob:
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    staging_dir.mkdir(parents=True, exist_ok=True)
    job_dir = staging_dir / job_folder_name(video)
    manifest_path = job_dir / MANIFEST_NAME

    if manifest_path.exists():
        existing = VideoFaceJob.load(manifest_path)
        if existing.source_path.resolve() != video:
            raise RuntimeError(f"Proxy job collision at {job_dir}")
        if not existing.source_is_unchanged():
            raise RuntimeError(
                f"Source video changed after proxy creation; finalize or remove the old job: {job_dir}"
            )
        missing = [path for path in existing.frame_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Proxy job is incomplete; missing {len(missing)} frame(s): {job_dir}"
            )
        info = sampler.probe(video)
        return PreparedJob(existing, False, info)

    if job_dir.exists() and any(job_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to use non-empty directory without a valid manifest: {job_dir}"
        )
    job_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        info, extracted, temp_dir = sampler.extract_frames(
            video,
            sample_seconds=sample_seconds,
            max_frames=max_frames,
            max_dimension=max_dimension,
        )
        proxy_frames: list[ProxyFrame] = []
        for index, extracted_path in enumerate(extracted):
            timestamp = min(index * sample_seconds, info.duration_seconds)
            timestamp_ms = round(timestamp * 1000)
            filename = f"frame_{index + 1:06d}_t{timestamp_ms:012d}ms.jpg"
            destination = job_dir / filename
            shutil.move(str(extracted_path), destination)
            generated.append(destination)
            proxy_frames.append(ProxyFrame(filename, timestamp))

        stat = video.stat()
        job = VideoFaceJob(
            schema_version=MANIFEST_VERSION,
            job_id=job_id_for_video(video),
            source_video=str(video),
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            created_at=datetime.now(UTC).isoformat(),
            sample_seconds=sample_seconds,
            video_duration_seconds=info.duration_seconds,
            video_codec=info.codec,
            frames=tuple(proxy_frames),
            job_dir=job_dir,
        )
        payload = asdict(job)
        payload.pop("job_dir")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json", dir=job_dir
        )
        os.close(descriptor)
        temp_manifest = Path(temp_name)
        try:
            temp_manifest.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(temp_manifest, manifest_path)
        finally:
            temp_manifest.unlink(missing_ok=True)
        return PreparedJob(job, True, info)
    except Exception:
        for path in generated:
            path.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def discover_jobs(
    staging_dir: Path, sources: set[Path] | None = None
) -> list[VideoFaceJob]:
    if not staging_dir.is_dir():
        return []
    wanted = {source.resolve() for source in sources} if sources else None
    jobs: list[VideoFaceJob] = []
    for manifest in sorted(staging_dir.glob(f"*/{MANIFEST_NAME}")):
        job = VideoFaceJob.load(manifest)
        if wanted is None or job.source_path.resolve() in wanted:
            jobs.append(job)
    return jobs
