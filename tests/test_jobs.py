from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from digikam_video_tagger import jobs
from digikam_video_tagger.jobs import (
    VideoFaceJob,
    completed_video_for_source,
    job_folder_name,
    job_id_for_video,
    mark_job_completed,
    staging_apply_lock,
)


def test_job_id_is_stable_and_case_insensitive_on_windows(tmp_path: Path) -> None:
    video = tmp_path / "Family Video.mp4"
    video.write_bytes(b"video")

    assert job_id_for_video(video) == job_id_for_video(Path(str(video).upper()))
    assert job_folder_name(video).startswith("Family_Video__")


def test_completion_ledger_matches_only_unchanged_source(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    job_dir = staging / "job"
    job_dir.mkdir(parents=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    stat = video.stat()
    job = VideoFaceJob(
        schema_version=1,
        job_id=job_id_for_video(video),
        source_video=str(video.resolve()),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        created_at="2026-01-01T00:00:00+00:00",
        sample_seconds=5.0,
        video_duration_seconds=1.0,
        video_codec="h264",
        frames=(),
        job_dir=job_dir,
    )

    mark_job_completed(job, ["People/Family/Shelby"], Path(f"{video}.xmp"))

    completed = completed_video_for_source(staging, video)
    assert completed is not None
    assert completed.people == ("People/Family/Shelby",)

    video.write_bytes(b"changed")
    assert completed_video_for_source(staging, video) is None


def test_completion_ledger_supports_reviewed_video_without_people(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    job_dir = staging / "job"
    job_dir.mkdir(parents=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    stat = video.stat()
    job = VideoFaceJob(
        schema_version=1,
        job_id=job_id_for_video(video),
        source_video=str(video.resolve()),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        created_at="2026-01-01T00:00:00+00:00",
        sample_seconds=5.0,
        video_duration_seconds=1.0,
        video_codec="h264",
        frames=(),
        job_dir=job_dir,
    )

    mark_job_completed(job, [], None)

    completed = completed_video_for_source(staging, video)
    assert completed is not None
    assert completed.people == ()
    assert completed.sidecar is None


def test_manifest_rejects_frame_path_outside_job_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    manifest = job_dir / ".digikam-video-face-job.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "job",
                "source_video": str(tmp_path / "video.mp4"),
                "source_size": 1,
                "source_mtime_ns": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "sample_seconds": 5.0,
                "video_duration_seconds": 1.0,
                "video_codec": "h264",
                "frames": [{"filename": "../outside.jpg", "timestamp_seconds": 0.0}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe proxy frame filename"):
        VideoFaceJob.load(manifest)


def test_version_one_completion_ledger_loads_with_safe_ownership_defaults(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / jobs.COMPLETED_NAME
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "videos": {
                    "abc": {
                        "source_video": str(tmp_path / "video.mp4"),
                        "source_size": 1,
                        "source_mtime_ns": 2,
                        "applied_at": "2026-01-01T00:00:00+00:00",
                        "people": ["People/Mom"],
                        "sidecar": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    entry = jobs.load_completed_videos(tmp_path)["abc"]
    assert entry.workflow == "confirmed"
    assert entry.managed_placeholders == ()
    assert entry.managed_placeholder_root is None
    assert entry.cluster_store_id is None


def test_completion_round_trip_with_managed_placeholders(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    job_dir = staging / "job"
    job_dir.mkdir(parents=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    stat = video.stat()
    job = VideoFaceJob(
        schema_version=1,
        job_id=job_id_for_video(video),
        source_video=str(video.resolve()),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        created_at="2026-01-01T00:00:00+00:00",
        sample_seconds=5.0,
        video_duration_seconds=1.0,
        video_codec="h264",
        frames=(),
        job_dir=job_dir,
    )

    mark_job_completed(
        job,
        ["People/Mom"],
        Path(f"{video}.xmp"),
        workflow="autofinalize",
        managed_placeholders=("People/Unknown/Person_001",),
        managed_placeholder_root="People/Unknown",
        cluster_store_id="store-123",
    )

    completed = completed_video_for_source(staging, video)
    assert completed is not None
    assert completed.workflow == "autofinalize"
    assert completed.managed_placeholders == ("People/Unknown/Person_001",)
    assert completed.managed_placeholder_root == "People/Unknown"
    assert completed.cluster_store_id == "store-123"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows apply lock",
)
def test_second_apply_lock_raises_while_first_is_held(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    with (
        staging_apply_lock(staging),
        pytest.raises(RuntimeError, match="Another apply operation"),
        staging_apply_lock(staging),
    ):
        pass


def test_apply_lock_unsupported_on_non_windows(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows only")
    with (
        pytest.raises(RuntimeError, match="only implemented on Windows"),
        staging_apply_lock(tmp_path / "staging"),
    ):
        pass
