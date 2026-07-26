# digiKam Video Tagger

Tag video files through digiKam without modifying the video bytes or digiKam's database. All metadata is written atomically to adjacent XMP sidecars such as `video.mp4.xmp`.

## Choose a workflow

| Goal | Command workflow |
|---|---|
| Copy **confirmed digiKam People** tags to videos | `prepare` → digiKam face review → `status` → `finalize --apply` |
| Recognize **known people** and cluster unknown faces automatically | `autofinalize` |
| Add automatic object/face-presence tags | `tag --apply` |

The confirmed-People workflow is the normal workflow. The automatic `autofinalize` workflow recognizes people from the read-only SFace gallery, assigns stable `People/Unknown/Person_NNN` placeholders to accepted unknown faces, and resolves those placeholders later when a cluster centroid matches the gallery. The automatic `tag` workflow does **not** create or confirm digiKam People regions.

## One-time setup

### 1. Install and configure the project

```powershell
git clone https://github.com/devdanks/digikam-video-tagger.git
cd digikam-video-tagger
uv sync
Copy-Item config.example.ps1 config.local.ps1
# Edit config.local.ps1 only if your database, model, FFmpeg, or ExifTool settings differ.
. .\config.local.ps1
uv run digikam-video-tagger doctor
```

`config.local.ps1` is ignored by Git. **Dot-source it in every new PowerShell session** before running the tool, so its local backend and database settings are available.

FFmpeg and ExifTool are found from `PATH` automatically. Do not add example placeholder paths unless the tools are not on `PATH`.

### 2. Configure digiKam once

1. Enable **XMP sidecar reading** in digiKam's Metadata settings.
2. Add this automatically managed directory as a digiKam **Album Root**:

   ```text
   %LOCALAPPDATA%\digikam-video-tagger\staging
   ```

The staging location is fixed by the application. **Do not configure or choose a staging path.** It must be an Album Root so digiKam can catalog the temporary JPEG proxy frames used by `prepare` and `finalize`. The `autofinalize` workflow writes sidecars directly and does not need the staging Album Root, but it still uses the staging directory for proxy frames, the cluster store, and the completion ledger.

## Confirmed People workflow

Run these steps in order for every batch.

### 1. Prepare proxy frames

```powershell
. .\config.local.ps1
uv run digikam-video-tagger prepare "G:\Videos"
```

The command creates temporary JPEG frames in the managed staging directory. It does not change the source videos or write People tags yet.

### 2. Review faces in digiKam

In digiKam, open the managed staging Album Root and:

1. Scan for new items, including subalbums.
2. Run face detection and recognition.
3. Confirm or assign People names.
4. Click **Apply** in digiKam.

### 3. Check readiness — do not skip this step

```powershell
. .\config.local.ps1
uv run digikam-video-tagger status "G:\Videos"
```

Only continue when the summary reports **`0 pending`** and **`0 failed`**. A pending job means one or more proxy frames are not yet catalogued by digiKam. Add/rescan the managed staging Album Root, then run `status` again.

### 4. Finalize the complete batch

```powershell
. .\config.local.ps1
uv run digikam-video-tagger finalize "G:\Videos" --apply
```

`--apply` is a total-completion operation:

- writes adjacent XMP sidecars for videos with confirmed People tags;
- records fully catalogued videos with no confirmed People tags as reviewed;
- removes all manifest-owned proxy frames and proxy job folders;
- refuses to remove a video’s proxies if any of its frames are not catalogued.

Afterward, reread metadata for the source-video album in digiKam.

### 5. Verify the written sidecars with ExifTool

Run this standard PowerShell command to inspect the People fields in every XMP sidecar under the
video root:

```powershell
# Uses the configured path when config.local.ps1 is loaded; otherwise uses PATH.
$exiftool = if ($env:DIGIKAM_VIDEO_TAGGER_EXIFTOOL) {
  $env:DIGIKAM_VIDEO_TAGGER_EXIFTOOL
} else {
  (Get-Command exiftool -ErrorAction Stop).Source
}
Get-ChildItem -LiteralPath 'G:\Videos' -Recurse -File -Filter '*.xmp' |
  ForEach-Object {
    & $exiftool -G1 -s -FileName `
      -XMP-digiKam:TagsList `
      -XMP-lr:HierarchicalSubject `
      -XMP-dc:Subject `
      $_.FullName
  }
```

For each person, the output should agree across all three fields. For example:

```text
[XMP-digiKam] TagsList            : People/Family/Shelby
[XMP-lr]      HierarchicalSubject : People|Family|Shelby
[XMP-dc]      Subject             : Shelby
```

## Automated People workflow (`autofinalize`)

`autofinalize` discovers videos, extracts proxy frames, recognizes people against the read-only
SFace gallery, and clusters unknown faces into stable `People/Unknown/Person_NNN` placeholders. It
writes adjacent XMP sidecars, records completion, and removes only the manifest-owned proxy frames.

```powershell
# Preview proposed tags without writing anything.
uv run digikam-video-tagger autofinalize "G:\Videos"

# Write sidecars, save accepted clusters, record completion, and clean proxies.
uv run digikam-video-tagger autofinalize "G:\Videos" --apply
```

On later runs, if an unknown cluster's centroid becomes a strong match for a newly trained
gallery identity, `autofinalize --apply` adds `People/<name>` to every unchanged source video that
owns that placeholder and removes the exact `People/Unknown/Person_NNN` tag from the sidecar.
Replacement only happens when the completion ledger proves the tool wrote the placeholder, so user
and unrelated tags are never touched.

Placeholder stability is best-effort, not biometric certainty. Before making `autofinalize` your
default workflow, run a local calibration set (repeated people, similar relatives, profiles,
occlusion, low light, motion blur, multiple people per frame, and false-positive detections) and
choose a conservative `--cluster-distance` that favors splitting one person into two placeholders
over merging two people into one.

## Automatic tags (`tag`)

Use `tag` only when you want automatic object or face-presence tags rather than the reviewed People workflow:

```powershell
# Preview only — writes nothing.
uv run digikam-video-tagger tag "G:\Videos"

# Write Auto Tags/Video sidecars.
uv run digikam-video-tagger tag "G:\Videos" --apply
```

`tag` uses YOLOv11 for objects and YuNet/SFace for faces. It writes object tags under `Auto Tags/Video/Objects/...`, adds `Auto Tags/Video/Contains Faces` when appropriate, and can add recognized `People/<name>` labels from the SFace gallery. It does not create digiKam face regions or replace manual face confirmation.

## Safety guarantees

- digiKam database access is read-only.
- Videos are never rewritten.
- Metadata is merged into `video.ext.xmp` sidecars atomically.
- Existing sidecar and embedded tags are preserved and deduplicated.
- `finalize --apply` deletes only files listed in its validated proxy-job manifests.
- A changed source video is not finalized from stale proxy frames.
- `autofinalize --apply` deletes only manifest-owned proxy frames and only removes exact
  `People/Unknown/Person_NNN` tags that the completion ledger proves the tool wrote.
- `autofinalize` persists a cluster store (`.digikam-video-face-clusters.json`) keyed to the SFace
  model fingerprint; a changed model requires an explicit rebuild and is never silently reused.
- Only one `autofinalize --apply` operation may run against a staging directory at a time.

## Development

```powershell
uv run pytest
uv run python -m compileall -q src tests
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). For digiKam configuration, see its documentation on [metadata sidecars](https://docs.digikam.org/en/setup_application/metadata_settings.html) and [database behavior](https://docs.digikam.org/en/getting_started/database_intro.html).
