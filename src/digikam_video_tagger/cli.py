from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_NAME,
    DEFAULT_DB_PASSWORD,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    DEFAULT_DIGIKAM_CONFIG,
    DEFAULT_EXIFTOOL,
    DEFAULT_FFMPEG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_STAGING_DIR,
    VIDEO_EXTENSIONS,
    DatabaseConfig,
    ToolPaths,
    digikam_sidecar_reading_enabled,
)
from .digikam_db import DigiKamCatalog, DigiKamFaceGallery
from .ffmpeg import FFmpegSampler
from .jobs import (
    MANIFEST_NAME,
    completed_video_for_source,
    discover_jobs,
    job_folder_name,
    load_completed_videos,
    mark_job_completed,
    prepare_job,
)
from .metadata import ExifToolSidecarWriter
from .models import FaceTagger, YoloObjectTagger, select_opencv_target
from .pipeline import AnalysisResult, VideoTaggingPipeline
from .process import run_command


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _tool_paths(args: argparse.Namespace) -> ToolPaths:
    return ToolPaths(args.ffmpeg_dir, args.exiftool, args.model_dir)


def _database_config(args: argparse.Namespace) -> DatabaseConfig:
    return DatabaseConfig(
        args.db_host, args.db_port, args.db_user, args.db_password, args.db_name
    )


def _require_cuda(args: argparse.Namespace) -> bool:
    return args.ffmpeg_cuda and not args.allow_cpu_fallback


def _require_opencl(args: argparse.Namespace) -> bool:
    return args.opencl and not args.allow_cpu_fallback


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ffmpeg-dir", type=_path, default=DEFAULT_FFMPEG_DIR)
    parser.add_argument("--exiftool", type=_path, default=DEFAULT_EXIFTOOL)
    parser.add_argument("--model-dir", type=_path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-password", default=DEFAULT_DB_PASSWORD)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Deprecated alias that permits CPU fallback for both FFmpeg and OpenCV",
    )
    parser.add_argument(
        "--ffmpeg-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA frame decoding (default: enabled; use --no-ffmpeg-cuda for CPU)",
    )
    parser.add_argument(
        "--opencl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require OpenCL DNN inference (default: enabled; use --no-opencl for CPU)",
    )


def _discover_videos(paths: list[Path], recursive: bool) -> list[Path]:
    videos: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS:
            videos.add(path.resolve())
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            videos.update(
                item.resolve()
                for item in iterator
                if item.is_file() and item.suffix.casefold() in VIDEO_EXTENSIONS
            )
    return sorted(videos, key=lambda item: str(item).casefold())


def _completed_count(staging_dir: Path, selected_sources: set[Path] | None) -> int:
    entries = [
        entry
        for entry in load_completed_videos(staging_dir).values()
        if entry.source_is_unchanged()
    ]
    if selected_sources is None:
        return len(entries)
    selected_keys = {str(source.resolve()).casefold() for source in selected_sources}
    return sum(
        1
        for entry in entries
        if str(entry.source_path.resolve()).casefold() in selected_keys
    )


def doctor(args: argparse.Namespace) -> int:
    paths = _tool_paths(args)
    checks: list[tuple[str, bool, str]] = []
    warnings: list[tuple[str, str]] = []
    object_model = paths.yolo_xlarge if args.object_model == "xl" else paths.yolo_nano
    for name, path in (
        ("ffmpeg", paths.ffmpeg),
        ("ffprobe", paths.ffprobe),
        ("ExifTool", paths.exiftool),
        ("YuNet", paths.yunet),
        ("SFace", paths.sface),
        (f"YOLOv11 {args.object_model}", object_model),
        ("COCO names", paths.coco_names),
    ):
        checks.append((name, path.is_file(), str(path)))

    sampler = FFmpegSampler(
        paths.ffmpeg, paths.ffprobe, require_cuda=_require_cuda(args)
    )
    try:
        version_output = run_command(
            [paths.ffmpeg, "-hide_banner", "-version"], timeout=15
        ).stdout
        version = version_output.splitlines()[0]
        checks.append(
            ("FFmpeg build", "--enable-gpl" in version_output.casefold(), version)
        )
        if _require_cuda(args):
            hwaccels = run_command(
                [paths.ffmpeg, "-hide_banner", "-hwaccels"], timeout=15
            ).stdout.casefold()
            checks.append(
                ("FFmpeg CUDA", "cuda" in hwaccels, "CUDA is listed by -hwaccels")
            )
            sampler.cuda_smoke_test()
            checks.append(
                ("CUDA device", True, "CUDA device 0 initialized and processed a frame")
            )
        else:
            checks.append(("FFmpeg CUDA", True, "not required"))
            checks.append(("CUDA device", True, "not required"))
    except Exception as error:
        checks.append(("CUDA device", False, str(error)))

    try:
        target = select_opencv_target(require_opencl=_require_opencl(args))
        checks.append(
            ("OpenCV DNN", True, f"OpenCV {cv2.__version__}, target={target.name}")
        )
        face_tagger = FaceTagger(paths.yunet, paths.sface, target, [])
        object_tagger = YoloObjectTagger(object_model, paths.coco_names, target)
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        face_tagger.detect(test_image)
        object_tagger.detect(test_image)
        checks.append(
            ("digiKam models", True, "YuNet, SFace, and YOLOv11 inference completed")
        )
    except Exception as error:
        checks.append(("digiKam models", False, str(error)))

    try:
        version = run_command([paths.exiftool, "-ver"], timeout=15).stdout.strip()
        checks.append(("ExifTool runtime", True, f"version {version}"))
    except Exception as error:
        checks.append(("ExifTool runtime", False, str(error)))

    sidecar_reading = digikam_sidecar_reading_enabled(DEFAULT_DIGIKAM_CONFIG)
    checks.append(
        (
            "digiKam sidecar reading",
            sidecar_reading is True,
            f"{DEFAULT_DIGIKAM_CONFIG}: "
            + ("enabled" if sidecar_reading is True else "disabled or not configured"),
        )
    )

    try:
        regions, person_tags, embeddings = DigiKamCatalog(
            _database_config(args)
        ).face_statistics()
        detail = (
            f"read-only connection succeeded; {regions} confirmed regions, "
            f"{person_tags} person tags, {embeddings} SFace embeddings"
        )
        checks.append(("digiKam face catalog", True, detail))
        if embeddings == 0:
            warnings.append(
                (
                    "digiKam recognition training",
                    "no SFace embeddings are stored; face detection and manual naming work, "
                    "but automatic name suggestions require digiKam's Retrain Faces operation",
                )
            )
    except Exception as error:
        checks.append(("digiKam face catalog", False, str(error)))

    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    for name, detail in warnings:
        print(f"[WARN] {name}: {detail}")
    failed = [name for name, passed, _ in checks if not passed]
    if failed:
        print(f"\nDoctor failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll required backends are ready.")
    return 0


def prepare(args: argparse.Namespace) -> int:
    paths = _tool_paths(args)
    videos = _discover_videos(args.paths, args.recursive)
    if not videos:
        print("No supported video files found.", file=sys.stderr)
        return 2

    sampler = FFmpegSampler(
        paths.ffmpeg, paths.ffprobe, require_cuda=_require_cuda(args)
    )
    total = len(videos)
    prepared = 0
    existing = 0
    completed = 0
    proxy_frames = 0
    failures = 0
    warnings = 0
    if not args.json:
        print(f"Discovered {total} video(s); directories are recursive by default.")

    for index, video in enumerate(videos, start=1):
        try:
            active_manifest = args.staging_dir / job_folder_name(video) / MANIFEST_NAME
            completion = None
            if not args.reprocess_completed and not active_manifest.is_file():
                completion = completed_video_for_source(args.staging_dir, video)
            if completion is not None:
                completed += 1
                payload = {
                    "type": "video",
                    "index": index,
                    "total": total,
                    "source_video": str(video),
                    "job_id": completion.job_id,
                    "completed": True,
                    "people": list(completion.people),
                    "sidecar": completion.sidecar,
                }
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False))
                elif not args.summary_only:
                    print(f"[{index}/{total} COMPLETED] {video}")
                continue

            result = prepare_job(
                video,
                args.staging_dir,
                sampler,
                sample_seconds=args.sample_seconds,
                max_frames=args.max_frames,
                max_dimension=args.max_dimension,
            )
            payload = {
                "type": "video",
                "index": index,
                "total": total,
                "source_video": str(video),
                "job_id": result.job.job_id,
                "job_dir": str(result.job.job_dir),
                "frames": len(result.job.frames),
                "created": result.created,
                "duration_seconds": result.info.duration_seconds,
                "codec": result.info.codec,
            }
            expected_frames = min(
                args.max_frames,
                max(1, math.ceil(result.info.duration_seconds / args.sample_seconds)),
            )
            sample_warning = None
            if len(result.job.frames) < expected_frames - 1:
                warnings += 1
                sample_warning = (
                    f"decoded timeline produced {len(result.job.frames)} frame(s), but the "
                    f"declared duration suggests about {expected_frames}; the source may have "
                    "a truncated or inaccurate video timeline"
                )
                payload["warning"] = sample_warning
            proxy_frames += len(result.job.frames)
            if result.created:
                prepared += 1
            else:
                existing += 1
            if args.json:
                print(json.dumps(payload, ensure_ascii=False))
            elif not args.summary_only:
                state = "PREPARED" if result.created else "EXISTS"
                print(f"[{index}/{total} {state}] {video}")
                print(f"  frames={len(result.job.frames)}, codec={result.info.codec}")
                print(f"  album={result.job.job_dir}")
                if sample_warning:
                    print(f"  warning={sample_warning}", file=sys.stderr)
        except Exception as error:
            failures += 1
            print(f"[{index}/{total} ERROR] {video}: {error}", file=sys.stderr)

    summary = {
        "type": "summary",
        "command": "prepare",
        "videos": total,
        "prepared": prepared,
        "existing": existing,
        "completed": completed,
        "failed": failures,
        "warnings": warnings,
        "proxy_frames": proxy_frames,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(
            "\nPrepare summary: "
            f"{total} video(s), {prepared} prepared, {existing} active, "
            f"{completed} previously completed, "
            f"{failures} failed, {warnings} warning(s), {proxy_frames} proxy frame(s)."
        )

    if not args.json and not failures and (prepared or existing):
        print("\nNext in digiKam:")
        print(f"  1. Scan the collection so {args.staging_dir} appears as an album.")
        print(
            "  2. Run normal face detection/recognition on that album and all sub-albums."
        )
        print("  3. Confirm or assign the person names and click Apply.")
        print("  4. Run this tool's finalize command for the same video(s).")
    return 1 if failures else 0


def finalize(args: argparse.Namespace) -> int:
    if getattr(args, "complete_without_people", False) and not args.apply:
        print("--complete-without-people requires --apply", file=sys.stderr)
        return 2

    selected_sources: set[Path] | None = None
    if args.paths:
        discovered = _discover_videos(args.paths, args.recursive)
        selected_sources = set(discovered)
        if not selected_sources:
            print("No supported source videos found.", file=sys.stderr)
            return 2

    jobs = discover_jobs(args.staging_dir, selected_sources)
    if not jobs:
        if getattr(args, "status_mode", False):
            completed_count = _completed_count(args.staging_dir, selected_sources)
            summary = {
                "type": "summary",
                "command": "status",
                "jobs": 0,
                "ready": 0,
                "applied": 0,
                "completed": completed_count,
                "pending": 0,
                "pending_unscanned": 0,
                "pending_uncatalogued": 0,
                "pending_unnamed": 0,
                "failed": 0,
            }
            if args.json:
                print(json.dumps(summary, ensure_ascii=False))
            else:
                print(
                    "Batch summary: "
                    f"0 active job(s), {completed_count} completed, 0 ready, "
                    "0 pending, 0 failed."
                )
            return 0
        print(f"No proxy jobs found under {args.staging_dir}.", file=sys.stderr)
        return 2

    catalog = DigiKamCatalog(_database_config(args))
    writer = ExifToolSidecarWriter(args.exiftool)
    total = len(jobs)
    failures = 0
    ready = 0
    applied = 0
    completed_without_people = 0
    pending_unscanned = 0
    pending_uncatalogued = 0
    pending_unnamed = 0
    if not args.json:
        print(f"Found {total} proxy job(s) in the batch.")

    for index, job in enumerate(jobs, start=1):
        try:
            if not job.source_is_unchanged():
                raise RuntimeError(
                    "Source video is missing or changed since proxy extraction"
                )
            missing_files = [frame for frame in job.frame_paths if not frame.is_file()]
            if missing_files:
                raise RuntimeError(
                    f"Proxy job is missing {len(missing_files)} generated frame(s)"
                )

            frame_faces = catalog.confirmed_faces_for_frames(job.frame_paths)
            not_scanned = [item.frame for item in frame_faces if item.image_id is None]
            not_catalogued = [
                item.frame
                for item in frame_faces
                if not getattr(item, "catalogued", True)
            ]
            people = sorted(
                {tag for item in frame_faces for tag in item.person_tag_paths},
                key=str.casefold,
            )
            payload = {
                "type": "video",
                "index": index,
                "total": total,
                "source_video": str(job.source_path),
                "job_dir": str(job.job_dir),
                "frames": len(job.frames),
                "catalogued_frames": len(job.frames) - len(not_scanned),
                "confirmed_people": people,
                "ready": not not_scanned and bool(people),
                "applied": False,
            }

            if not_scanned:
                if not_catalogued:
                    pending_uncatalogued += 1
                else:
                    pending_unscanned += 1
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False))
                elif not args.summary_only:
                    if not_catalogued:
                        print(
                            f"[{index}/{total} STAGING-NOT-CATALOGUED] {job.source_path}"
                        )
                        print(
                            "  staging directory is not under any digiKam Album Root; add it as a "
                            "collection root and rescan"
                        )
                    else:
                        print(f"[{index}/{total} PENDING] {job.source_path}")
                        print(
                            f"  {len(not_scanned)} of {len(job.frames)} proxy frames are not in digiKam yet"
                        )
                continue
            if not people:
                if args.apply and getattr(args, "complete_without_people", False):
                    mark_job_completed(job, [], None)
                    removed = [] if args.keep_frames else job.remove_generated_files()
                    payload["completed_without_people"] = True
                    payload["removed_proxy_files"] = len(removed)
                    completed_without_people += 1
                    if args.json:
                        print(json.dumps(payload, ensure_ascii=False))
                    elif not args.summary_only:
                        print(
                            f"[{index}/{total} COMPLETED-NO-PEOPLE] {job.source_path}"
                        )
                        if not args.keep_frames:
                            print(
                                "  generated proxy frames deleted; no video sidecar was needed"
                            )
                else:
                    pending_unnamed += 1
                    if args.json:
                        print(json.dumps(payload, ensure_ascii=False))
                    elif not args.summary_only:
                        print(f"[{index}/{total} PENDING] {job.source_path}")
                        print(
                            "  proxy frames are catalogued, but no confirmed person face regions were found"
                        )
                continue

            ready += 1
            if args.apply:
                metadata = writer.write_tags(job.source_path, people)
                payload["applied"] = True
                payload["sidecar"] = str(metadata.sidecar)
                payload["added_tags"] = list(metadata.added_tags)
                mark_job_completed(job, people, metadata.sidecar)
                removed = [] if args.keep_frames else job.remove_generated_files()
                payload["removed_proxy_files"] = len(removed)
                state = "APPLIED"
                applied += 1
            else:
                state = "READY"

            if args.json:
                print(json.dumps(payload, ensure_ascii=False))
            elif not args.summary_only:
                print(f"[{index}/{total} {state}] {job.source_path}")
                print(f"  confirmed_people={', '.join(people)}")
                if args.apply:
                    print(f"  sidecar={payload['sidecar']}")
                    if not args.keep_frames:
                        print(
                            "  generated proxy frames deleted; rescan the staging album in digiKam"
                        )
        except Exception as error:
            failures += 1
            print(
                f"[{index}/{total} ERROR] {job.source_path}: {error}", file=sys.stderr
            )

    pending = pending_unscanned + pending_uncatalogued + pending_unnamed
    completed_count = _completed_count(args.staging_dir, selected_sources)
    summary = {
        "type": "summary",
        "command": "finalize" if args.apply else "status",
        "jobs": total,
        "ready": ready,
        "applied": applied,
        "completed_without_people": completed_without_people,
        "completed": completed_count,
        "pending": pending,
        "pending_unscanned": pending_unscanned,
        "pending_uncatalogued": pending_uncatalogued,
        "pending_unnamed": pending_unnamed,
        "failed": failures,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        if args.apply:
            prefix = f"{total} job(s) checked, {completed_count} completed, {applied} applied"
            if completed_without_people:
                prefix += f", {completed_without_people} completed without people"
        else:
            prefix = (
                f"{total} active job(s), {completed_count} completed, {ready} ready"
            )
        print(
            "\nBatch summary: "
            f"{prefix}, {pending} pending "
            f"({pending_unscanned} unscanned, {pending_uncatalogued} outside collection roots, "
            f"{pending_unnamed} without confirmed people), "
            f"{failures} failed."
        )

    # Pending jobs are an expected partial-batch state: ready jobs can be
    # applied while videos without confirmed people remain for later review.
    return 1 if failures else 0


def _result_payload(result: AnalysisResult) -> dict:
    return {
        "video": str(result.video),
        "duration_seconds": result.info.duration_seconds,
        "codec": result.info.codec,
        "frames": result.frame_count,
        "unreadable_frames": result.unreadable_frames,
        "face_frames": result.face_frames,
        "objects": [asdict(item) for item in result.objects],
        "people": [asdict(item) for item in result.people],
        "tags": list(result.tags),
        "sidecar": str(result.metadata.sidecar) if result.metadata else None,
        "added_tags": list(result.metadata.added_tags) if result.metadata else [],
    }


def tag(args: argparse.Namespace) -> int:
    paths = _tool_paths(args)
    videos = _discover_videos(args.paths, args.recursive)
    if not videos:
        print("No supported video files found.", file=sys.stderr)
        return 2

    target = select_opencv_target(require_opencl=_require_opencl(args))
    gallery = (
        [] if args.no_people else DigiKamFaceGallery(_database_config(args)).load()
    )
    object_tagger = None
    if not args.no_objects:
        object_model = (
            paths.yolo_xlarge if args.object_model == "xl" else paths.yolo_nano
        )
        object_tagger = YoloObjectTagger(
            object_model,
            paths.coco_names,
            target,
            confidence_threshold=args.object_confidence,
        )
    face_tagger = None
    if not args.no_faces:
        face_tagger = FaceTagger(
            paths.yunet,
            paths.sface,
            target,
            gallery,
            detection_threshold=args.face_confidence,
            recognition_distance=args.person_distance,
        )

    pipeline = VideoTaggingPipeline(
        FFmpegSampler(paths.ffmpeg, paths.ffprobe, require_cuda=_require_cuda(args)),
        object_tagger,
        face_tagger,
        ExifToolSidecarWriter(paths.exiftool),
        tag_root=args.tag_root,
        sample_seconds=args.sample_seconds,
        max_frames=args.max_frames,
        max_dimension=args.max_dimension,
        min_object_hits=args.min_object_hits,
        min_person_hits=args.min_person_hits,
        min_frame_ratio=args.min_frame_ratio,
        max_object_tags=args.max_object_tags,
    )

    failures = 0
    for video in videos:
        try:
            result = pipeline.analyze(video, apply=args.apply)
            payload = _result_payload(result)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False))
            else:
                mode = "APPLIED" if args.apply else "DRY-RUN"
                print(f"[{mode}] {video}")
                print(
                    f"  frames={result.frame_count}, unreadable_frames={result.unreadable_frames}, "
                    f"face_frames={result.face_frames}, codec={result.info.codec}"
                )
                print(f"  tags={', '.join(result.tags) if result.tags else '(none)'}")
                if result.metadata:
                    print(f"  sidecar={result.metadata.sidecar}")
        except Exception as error:
            failures += 1
            print(f"[ERROR] {video}: {error}", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="digikam-video-tagger")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Verify every runtime prerequisite"
    )
    _add_shared_options(doctor_parser)
    doctor_parser.add_argument("--object-model", choices=("nano", "xl"), default="nano")
    doctor_parser.set_defaults(handler=doctor)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract temporary proxy frames for digiKam's native People workflow",
    )
    _add_shared_options(prepare_parser)
    prepare_parser.add_argument(
        "paths",
        nargs="+",
        type=_path,
        help="Video file(s) or directory tree(s); directories recurse by default",
    )
    prepare_parser.add_argument(
        "--staging-dir", type=_path, default=DEFAULT_STAGING_DIR
    )
    prepare_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse through every supplied directory (default: enabled)",
    )
    prepare_parser.add_argument("--sample-seconds", type=float, default=5.0)
    prepare_parser.add_argument("--max-frames", type=int, default=120)
    prepare_parser.add_argument("--max-dimension", type=int, default=1920)
    prepare_parser.add_argument(
        "--reprocess-completed",
        action="store_true",
        help="Regenerate proxies even when an unchanged video is in the completion ledger",
    )
    prepare_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress per-video success lines and print only batch totals",
    )
    prepare_parser.add_argument("--json", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Copy confirmed proxy-frame People tags to source videos",
    )
    _add_shared_options(finalize_parser)
    finalize_parser.add_argument(
        "paths",
        nargs="*",
        type=_path,
        help="Optional source file(s) or directory tree(s); omit to process all jobs",
    )
    finalize_parser.add_argument(
        "--staging-dir", type=_path, default=DEFAULT_STAGING_DIR
    )
    finalize_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse through every supplied directory (default: enabled)",
    )
    finalize_parser.add_argument(
        "--apply", action="store_true", help="Write video XMP sidecars"
    )
    finalize_parser.add_argument(
        "--complete-without-people",
        action="store_true",
        help=(
            "With --apply, record fully scanned jobs with no confirmed People regions as "
            "reviewed and remove their generated frames"
        ),
    )
    finalize_parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep generated proxy frames after successful application",
    )
    finalize_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress per-video detail and print only aggregate batch state",
    )
    finalize_parser.add_argument("--json", action="store_true")
    finalize_parser.set_defaults(handler=finalize, status_mode=False)

    status_parser = subparsers.add_parser(
        "status",
        help="Summarize every recursive proxy job without writing or deleting anything",
    )
    _add_shared_options(status_parser)
    status_parser.add_argument(
        "paths",
        nargs="*",
        type=_path,
        help="Optional source file(s) or directory tree(s); omit to summarize all jobs",
    )
    status_parser.add_argument("--staging-dir", type=_path, default=DEFAULT_STAGING_DIR)
    status_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse through every supplied directory (default: enabled)",
    )
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(
        handler=finalize,
        apply=False,
        complete_without_people=False,
        keep_frames=True,
        summary_only=True,
        status_mode=True,
    )

    tag_parser = subparsers.add_parser(
        "tag",
        help="Experimental direct object/face-presence tagging (does not populate People regions)",
    )
    _add_shared_options(tag_parser)
    tag_parser.add_argument("paths", nargs="+", type=_path)
    tag_parser.add_argument(
        "--apply", action="store_true", help="Write merge-only XMP sidecars"
    )
    tag_parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    tag_parser.add_argument(
        "--json", action="store_true", help="Emit one JSON object per video"
    )
    tag_parser.add_argument("--sample-seconds", type=float, default=5.0)
    tag_parser.add_argument("--max-frames", type=int, default=120)
    tag_parser.add_argument("--max-dimension", type=int, default=1280)
    tag_parser.add_argument("--tag-root", default="Auto Tags/Video")
    tag_parser.add_argument("--object-model", choices=("nano", "xl"), default="nano")
    tag_parser.add_argument("--object-confidence", type=float, default=0.45)
    tag_parser.add_argument("--face-confidence", type=float, default=0.70)
    tag_parser.add_argument("--person-distance", type=float, default=0.50)
    tag_parser.add_argument("--min-object-hits", type=int, default=2)
    tag_parser.add_argument("--min-person-hits", type=int, default=2)
    tag_parser.add_argument("--min-frame-ratio", type=float, default=0.05)
    tag_parser.add_argument("--max-object-tags", type=int, default=20)
    tag_parser.add_argument("--no-objects", action="store_true")
    tag_parser.add_argument("--no-faces", action="store_true")
    tag_parser.add_argument("--no-people", action="store_true")
    tag_parser.set_defaults(handler=tag)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Batch commands are commonly piped through PowerShell or task runners, where
    # Python otherwise block-buffers progress until the process exits.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (OSError, ValueError):
                pass
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
