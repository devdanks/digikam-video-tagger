# digiKam Video Tagger

GPU-assisted People tagging for videos using digiKam's existing face-recognition workflow.

The tool extracts temporary JPEG proxy frames with FFmpeg, lets digiKam detect and confirm faces,
then copies the confirmed People hierarchy to adjacent video XMP sidecars. It never writes directly
to digiKam's database or rewrites the video stream.

> **Status:** Alpha. The supported workflow currently targets Windows, digiKam with MariaDB, and an
> NVIDIA FFmpeg build with CUDA decoding.

## How it works

```text
video tree -> CUDA proxy frames -> digiKam face workflow -> video.ext.xmp -> proxy cleanup
```

Each video receives an isolated, manifest-backed proxy folder. Folder inputs recurse by default,
completed videos are tracked, and reruns skip unchanged work. Sidecars preserve existing metadata
while adding digiKam-compatible `TagsList`, `HierarchicalSubject`, and `Subject` values.

## Requirements

- Windows with Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- digiKam with XMP sidecar reading enabled
- digiKam MariaDB reachable in read-only use
- FFmpeg and FFprobe with NVIDIA CUDA support
- ExifTool (the copy bundled with digiKam is suitable)
- digiKam face models and trained recognition data

## Install

```powershell
git clone https://github.com/devdanks/digikam-video-tagger.git
cd digikam-video-tagger
uv sync
```

FFmpeg is discovered from `PATH`. For the remaining local settings, copy and load the PowerShell
configuration template:

```powershell
Copy-Item config.example.ps1 config.local.ps1
# Edit config.local.ps1 for this workstation.
. .\config.local.ps1
uv run digikam-video-tagger doctor
```

`config.local.ps1` is ignored by Git. Every setting can also be supplied as a CLI option. Run
`uv run digikam-video-tagger --help` and the relevant subcommand's `--help` for details.

## Two-pass People workflow

Prepare an entire video tree:

```powershell
uv run digikam-video-tagger prepare "D:\Media\Videos"
```

In digiKam:

1. Scan for new items under `_digikam_video_faces`.
2. Detect and recognize faces on that album, including all subalbums.
3. Confirm or assign People names and click **Apply**.

Inspect the batch without writing or deleting anything:

```powershell
uv run digikam-video-tagger status "D:\Media\Videos"
```

Apply confirmed People tags and remove only successfully processed proxies:

```powershell
uv run digikam-video-tagger finalize "D:\Media\Videos" --apply
```

After reviewing the whole proxy batch, complete scanned videos that contain no confirmed people and
remove their temporary frames:

```powershell
uv run digikam-video-tagger finalize "D:\Media\Videos" `
  --apply --complete-without-people --summary-only
```

Reread metadata for the source-video album in digiKam afterward. Use `--keep-frames` to retain
generated JPEGs, or `prepare --reprocess-completed` to deliberately scan completed videos again.

## Safety

- digiKam database connections are query-only.
- Video bytes and timestamps are not modified.
- Metadata is merged into `filename.ext.xmp` sidecars atomically.
- Cleanup removes only manifest-owned proxy files and their sidecars.
- Jobs stop if a source video changes after extraction.

## Development

```powershell
uv run pytest
uv run python -m compileall -q src tests
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). digiKam's relevant documentation
covers [metadata sidecars](https://docs.digikam.org/en/setup_application/metadata_settings.html) and
[database behavior](https://docs.digikam.org/en/getting_started/database_intro.html).
