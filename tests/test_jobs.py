from pathlib import Path

from digikam_video_tagger.jobs import (
    VideoFaceJob,
    completed_video_for_source,
    job_folder_name,
    job_id_for_video,
    mark_job_completed,
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
