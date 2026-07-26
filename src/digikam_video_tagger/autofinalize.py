from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from .clustering import FaceClusterStore
from .evidence import EvidenceAccumulator
from .ffmpeg import FFmpegSampler
from .jobs import (
    CompletedVideo,
    PreparedJob,
    VideoFaceJob,
    discover_jobs,
    job_id_for_video,
    load_completed_videos,
    mark_job_completed,
    prepare_job,
)
from .metadata import ExifToolSidecarWriter, MetadataWriteResult
from .models import FaceDetection, FaceTagger
from .tags import people_tag


@dataclass(frozen=True)
class AutoFinalizeOptions:
    sample_seconds: float
    max_frames: int
    max_dimension: int
    min_person_hits: int
    min_frame_ratio: float


@dataclass(frozen=True)
class AutoFinalizeVideoResult:
    source_video: Path
    job_id: str
    frame_count: int
    unreadable_frames: int
    face_frames: int
    known_people: tuple[str, ...]
    placeholder_people: tuple[str, ...]
    proposed_replacements: tuple[tuple[str, str], ...]
    completed: bool
    applied: bool
    sidecar: Path | None
    removed_proxy_files: int
    error: str | None = None


@dataclass(frozen=True)
class AutoFinalizeSummary:
    videos: int
    applied: int
    completed: int
    known_people: int
    clustered_people: int
    resolved_people: int
    failed: int


class _DryRunFrames:
    def __init__(
        self,
        frames: list[Path],
        temp_dir: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        self.frames = frames
        self._temp_dir = temp_dir

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


class AutoFinalizeService:
    def __init__(
        self,
        staging_dir: Path,
        sampler: FFmpegSampler,
        face_tagger: FaceTagger,
        cluster_store: FaceClusterStore,
        cluster_store_path: Path,
        sidecar_writer: ExifToolSidecarWriter,
        options: AutoFinalizeOptions,
        *,
        prepare_job: Callable[..., PreparedJob] = prepare_job,
        discover_jobs: Callable[..., list[VideoFaceJob]] = discover_jobs,
        load_completed: Callable[
            [Path], dict[str, CompletedVideo]
        ] = load_completed_videos,
        mark_completed: Callable[..., CompletedVideo] = mark_job_completed,
        job_id: Callable[[Path], str] = job_id_for_video,
    ) -> None:
        if options.sample_seconds <= 0:
            raise ValueError("sample_seconds must be positive")
        if options.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if options.max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        if options.min_person_hits < 1:
            raise ValueError("min_person_hits must be positive")
        if not 0.0 <= options.min_frame_ratio <= 1.0:
            raise ValueError("min_frame_ratio must be between 0 and 1")

        self.staging_dir = staging_dir
        self.sampler = sampler
        self.face_tagger = face_tagger
        self.cluster_store = cluster_store
        self.cluster_store_path = cluster_store_path
        self.sidecar_writer = sidecar_writer
        self.options = options
        self.prepare_job = prepare_job
        self.discover_jobs = discover_jobs
        self.load_completed = load_completed
        self.mark_completed = mark_completed
        self.job_id = job_id

    def run(
        self,
        videos: list[Path],
        *,
        apply: bool,
        reprocess_completed: bool,
    ) -> tuple[list[AutoFinalizeVideoResult], AutoFinalizeSummary]:
        results: list[AutoFinalizeVideoResult] = []
        for video in videos:
            result = self._process_video(
                video, apply=apply, reprocess_completed=reprocess_completed
            )
            results.append(result)

        summary = AutoFinalizeSummary(
            videos=len(results),
            applied=sum(1 for r in results if r.applied),
            completed=sum(1 for r in results if r.completed),
            known_people=len({name for r in results for name in r.known_people}),
            clustered_people=len(
                {name for r in results for name in r.placeholder_people}
            ),
            resolved_people=sum(len(r.proposed_replacements) for r in results),
            failed=sum(1 for r in results if r.error is not None),
        )
        return results, summary

    def _process_video(
        self,
        video: Path,
        *,
        apply: bool,
        reprocess_completed: bool,
    ) -> AutoFinalizeVideoResult:
        video = video.resolve()
        job_id = self.job_id(video)
        active_job = self._active_job_for_video(video)

        if active_job is None and not reprocess_completed:
            completed = completed_video_for_source_with_loader(
                self.staging_dir, video, self.load_completed
            )
            if completed is not None:
                return AutoFinalizeVideoResult(
                    source_video=video,
                    job_id=job_id,
                    frame_count=0,
                    unreadable_frames=0,
                    face_frames=0,
                    known_people=completed.people,
                    placeholder_people=completed.managed_placeholders,
                    proposed_replacements=(),
                    completed=True,
                    applied=False,
                    sidecar=Path(completed.sidecar) if completed.sidecar else None,
                    removed_proxy_files=0,
                )

        try:
            if apply:
                prepared = self.prepare_job(
                    video,
                    self.staging_dir,
                    self.sampler,
                    sample_seconds=self.options.sample_seconds,
                    max_frames=self.options.max_frames,
                    max_dimension=self.options.max_dimension,
                )
                job = prepared.job
                frames = job.frame_paths
            else:
                dry_run = self._dry_run_frames(video, active_job)
                frames = dry_run.frames

            analysis = self._analyze_frames(frames)

            if not apply:
                dry_run = self._dry_run_frames(video, active_job)
                dry_run.cleanup()
                return self._build_dry_run_result(
                    video, job_id, analysis, cluster_session=None
                )

            return self._apply_analysis(video, job, analysis)
        except Exception as error:
            return AutoFinalizeVideoResult(
                source_video=video,
                job_id=job_id,
                frame_count=0,
                unreadable_frames=0,
                face_frames=0,
                known_people=(),
                placeholder_people=(),
                proposed_replacements=(),
                completed=False,
                applied=False,
                sidecar=None,
                removed_proxy_files=0,
                error=str(error),
            )

    def _active_job_for_video(self, video: Path) -> VideoFaceJob | None:
        jobs = self.discover_jobs(self.staging_dir, sources={video})
        return jobs[0] if jobs else None

    def _dry_run_frames(
        self, video: Path, active_job: VideoFaceJob | None
    ) -> _DryRunFrames:
        if active_job is not None:
            return _DryRunFrames(active_job.frame_paths, None)
        _info, frames, temp_dir = self.sampler.extract_frames(
            video,
            sample_seconds=self.options.sample_seconds,
            max_frames=self.options.max_frames,
            max_dimension=self.options.max_dimension,
        )
        return _DryRunFrames(frames, temp_dir)

    def _analyze_frames(
        self,
        frames: list[Path],
        *,
        cluster_session=None,
    ) -> _FrameAnalysis:
        if cluster_session is None:
            cluster_session = self.cluster_store.begin_session()
        evidence = EvidenceAccumulator()
        unreadable_frames = 0
        face_frames = 0
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable_frames += 1
                evidence.add_frame({})
                continue

            detections = self.face_tagger.detect_faces(image)
            frame_labels: dict[str, float] = {}
            if detections:
                face_frames += 1
            for detection in detections:
                token = self._token_for_detection(detection, cluster_session)
                frame_labels[token] = max(
                    frame_labels.get(token, 0.0), detection.confidence
                )
            evidence.add_frame(frame_labels)

        accepted = evidence.accepted(
            min_hits=self.options.min_person_hits,
            min_frame_ratio=self.options.min_frame_ratio,
        )
        known_tokens = [ev for ev in accepted if ev.label.startswith("known:")]
        unknown_tokens = {
            ev.label for ev in accepted if not ev.label.startswith("known:")
        }
        known_people = sorted(
            {ev.label.split(":", 1)[1] for ev in known_tokens},
            key=str.casefold,
        )
        placeholder_map = cluster_session.commit(unknown_tokens)
        placeholder_tags = sorted(placeholder_map.values(), key=str.casefold)

        return _FrameAnalysis(
            frame_count=len(frames),
            unreadable_frames=unreadable_frames,
            face_frames=face_frames,
            known_people=tuple(people_tag(name) for name in known_people),
            placeholder_people=tuple(placeholder_tags),
            store_changed=bool(unknown_tokens),
        )

    def _token_for_detection(self, detection: FaceDetection, session) -> str:
        if detection.name is not None:
            return f"known:{detection.name}"
        return session.assign(detection.embedding)

    def _build_dry_run_result(
        self,
        video: Path,
        job_id: str,
        analysis: _FrameAnalysis,
        cluster_session,
    ) -> AutoFinalizeVideoResult:
        return AutoFinalizeVideoResult(
            source_video=video,
            job_id=job_id,
            frame_count=analysis.frame_count,
            unreadable_frames=analysis.unreadable_frames,
            face_frames=analysis.face_frames,
            known_people=analysis.known_people,
            placeholder_people=analysis.placeholder_people,
            proposed_replacements=(),
            completed=False,
            applied=False,
            sidecar=None,
            removed_proxy_files=0,
        )

    def _apply_analysis(
        self,
        video: Path,
        job: VideoFaceJob,
        analysis: _FrameAnalysis,
    ) -> AutoFinalizeVideoResult:
        sidecar_path = self.sidecar_writer.sidecar_path(video)
        if analysis.store_changed:
            self.cluster_store.save(self.cluster_store_path)

        final_tags = list(analysis.known_people) + list(analysis.placeholder_people)
        write_result: MetadataWriteResult | None = None
        if final_tags:
            write_result = self.sidecar_writer.write_tags(video, final_tags)

        self.mark_completed(
            job,
            final_tags,
            sidecar_path if write_result is not None else None,
            workflow="autofinalize",
            managed_placeholders=analysis.placeholder_people,
            managed_placeholder_root=self.cluster_store.unknown_root,
            cluster_store_id=self.cluster_store.store_id,
        )
        removed = job.remove_generated_files()

        return AutoFinalizeVideoResult(
            source_video=video,
            job_id=job.job_id,
            frame_count=analysis.frame_count,
            unreadable_frames=analysis.unreadable_frames,
            face_frames=analysis.face_frames,
            known_people=analysis.known_people,
            placeholder_people=analysis.placeholder_people,
            proposed_replacements=(),
            completed=True,
            applied=True,
            sidecar=write_result.sidecar if write_result else sidecar_path,
            removed_proxy_files=len(removed),
        )


@dataclass(frozen=True)
class _FrameAnalysis:
    frame_count: int
    unreadable_frames: int
    face_frames: int
    known_people: tuple[str, ...]
    placeholder_people: tuple[str, ...]
    store_changed: bool


def completed_video_for_source_with_loader(
    staging_dir: Path,
    video: Path,
    loader: Callable[[Path], dict[str, CompletedVideo]],
) -> CompletedVideo | None:
    video = video.resolve()
    entry = loader(staging_dir).get(job_id_for_video(video))
    if (
        entry is None
        or entry.source_path.resolve() != video
        or not entry.source_is_unchanged()
    ):
        return None
    return entry
