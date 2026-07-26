# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 package using the `src` layout.

- `src/digikam_video_tagger/`: application code.
- `cli.py`: argparse command definitions and batch orchestration for all six subcommands
  (`doctor`, `prepare`, `status`, `finalize`, `tag`, `autofinalize`).
- `ffmpeg.py`: FFmpeg/FFprobe CUDA frame extraction with CPU fallback; `VideoInfo` probe.
- `pipeline.py`: `VideoTaggingPipeline` — the `tag` command's analysis flow (YOLOv11 objects +
  YuNet/SFace faces -> `EvidenceAccumulator` gating -> `Auto Tags/Video` sidecar tags).
- `models.py`: `YoloObjectTagger` (YOLOv11 ONNX), `FaceTagger` (YuNet detect + SFace recognize,
  now with gallery-independent `detect_faces()`), and `select_opencv_target` (OpenCL/CPU DNN target
  selection).
- `evidence.py`: `EvidenceAccumulator` and `TagEvidence` — per-label hit/frame-ratio gating used
  by `pipeline.py` and `autofinalize.py`.
- `clustering.py`: `FaceClusterStore` and `ClusterSession` — validated persistent unknown-face
  clusters, provisional per-video assignments, accepted-only commits, model-fingerprint compatibility,
  and known-person resolution candidates.
- `autofinalize.py`: `AutoFinalizeService` — the `autofinalize` command's per-video analysis,
  evidence gating, apply ordering, recovery, and ledger-owned placeholder replacement.
- `digikam_db.py`: read-only digiKam catalog access. `DigiKamCatalog` reads confirmed face
  assignments for `finalize`; `DigiKamFaceGallery` reads SFace training embeddings for `tag` and
  `autofinalize`.
- `metadata.py`: `ExifToolSidecarWriter` — atomic XMP sidecar merge (`TagsList`,
  `HierarchicalSubject`, `Subject`) and exact managed-tag removal via tempfile + `os.replace`.
- `jobs.py`: proxy job manifests, completion ledger (now version 2 with managed-placeholder
  ownership), manifest-owned cleanup, apply locking, and rerun skipping.
- `config.py`: environment-driven defaults, `ToolPaths`, `DatabaseConfig`, digiKamrc boolean reads.
- `process.py`: `run_command` — argv-form subprocess wrapper with `CREATE_NO_WINDOW`.
- `__init__.py`: OpenCL DNN kernel-cache setup and `__version__`.
- `tests/`: pytest tests named `test_<module>.py`, including `test_models.py`, `test_clustering.py`,
  and `test_autofinalize.py`.
- `README.md`: workstation setup, the confirmed-People workflow, the `autofinalize` workflow, and
  the direct `tag` auto-tagging workflow.
- `pyproject.toml` and `uv.lock`: package metadata and locked dependencies.

Keep generated proxy frames, model data, and local databases outside the repository.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
uv sync
uv run pytest
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv build
uv run digikam-video-tagger doctor
```

`uv sync` installs locked runtime and development dependencies. `pytest` runs the full suite. The
compile check catches syntax and import-time issues. `ruff check` and `ruff format --check` enforce
the project's lint and format rules. `uv build` creates distribution artifacts.
`doctor` validates FFmpeg/CUDA, OpenCV OpenCL, ExifTool, models, sidecar settings, and database access.

For safe workflow checks, prefer the read-only command:

```powershell
uv run digikam-video-tagger status
```

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 naming, explicit type hints, and `pathlib.Path` for filesystem
operations. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase
names for module constants. Keep CLI output actionable and preserve JSON output compatibility.
No formatter or linter is currently configured; match the existing style and keep imports grouped.

## Testing Guidelines

Use pytest and add focused regression tests for every bug fix. Name tests descriptively, such as
`test_finalize_can_complete_reviewed_job_without_people` or
`test_rejected_session_cluster_does_not_consume_id`. Mock external programs and digiKam catalog
access in unit tests; do not require the workstation database or GPU for the normal test suite.
All tests use fakes and temporary directories.

## Commit & Pull Request Guidelines

Use concise Conventional Commit subjects, for example `fix(ffmpeg): constrain portrait proxy
dimensions`. Pull requests should explain behavior changes,
data-safety implications, tests run, and any manual digiKam verification. Link relevant issues and
include CLI output when it clarifies a workflow change.

## Safety & Configuration

Never write directly to digiKam’s database. Treat it as read-only and propagate tags through
`filename.ext.xmp` sidecars. Do not commit personal paths, credentials, databases, models, generated
frames, or media. Keep workstation settings in ignored `config.local.ps1`. Cleanup must target only
files recorded by a validated job manifest. Managed placeholder tags may only be removed when the
completion ledger proves the tool wrote them.
