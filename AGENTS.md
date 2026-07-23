# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 package using the `src` layout.

- `src/digikam_video_tagger/`: application code.
- `cli.py`: command definitions and batch orchestration.
- `ffmpeg.py` and `pipeline.py`: GPU frame extraction and analysis flow.
- `digikam_db.py`: read-only digiKam catalog access.
- `metadata.py` and `jobs.py`: XMP sidecar writing, manifests, cleanup, and completion tracking.
- `tests/`: pytest tests named `test_<module>.py`.
- `README.md`: workstation setup and the two-pass People-tagging workflow.
- `pyproject.toml` and `uv.lock`: package metadata and locked dependencies.

Keep generated proxy frames, model data, and local databases outside the repository.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
uv sync
uv run pytest
uv run python -m compileall -q src tests
uv build
uv run digikam-video-tagger doctor
```

`uv sync` installs locked runtime and development dependencies. `pytest` runs the full suite. The
compile check catches syntax and import-time issues. `uv build` creates distribution artifacts.
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
`test_finalize_can_complete_reviewed_job_without_people`. Mock external programs and digiKam catalog
access in unit tests; do not require the workstation database or GPU for the normal test suite.

## Commit & Pull Request Guidelines

Use concise Conventional Commit subjects, for example `fix(ffmpeg): constrain portrait proxy
dimensions`. Pull requests should explain behavior changes,
data-safety implications, tests run, and any manual digiKam verification. Link relevant issues and
include CLI output when it clarifies a workflow change.

## Safety & Configuration

Never write directly to digiKam’s database. Treat it as read-only and propagate tags through
`filename.ext.xmp` sidecars. Do not commit personal paths, credentials, databases, models, generated
frames, or media. Keep workstation settings in ignored `config.local.ps1`. Cleanup must target only
files recorded by a validated job manifest.
