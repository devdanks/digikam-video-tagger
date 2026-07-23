# Copy this file to config.local.ps1, update the values, then dot-source it:
#   . .\config.local.ps1

$env:DIGIKAM_VIDEO_TAGGER_FFMPEG_DIR = 'C:\path\to\ffmpeg\bin'
$env:DIGIKAM_VIDEO_TAGGER_EXIFTOOL = 'C:\Program Files\digiKam\exiftool.exe'
$env:DIGIKAM_VIDEO_TAGGER_STAGING_DIR = 'D:\Pictures\_digikam_video_faces'
$env:DIGIKAM_VIDEO_TAGGER_MODEL_DIR = "$env:LOCALAPPDATA\digikam\facesengine"
$env:DIGIKAM_VIDEO_TAGGER_DIGIKAM_CONFIG = "$env:LOCALAPPDATA\digikamrc"

$env:DIGIKAM_VIDEO_TAGGER_DB_HOST = '127.0.0.1'
$env:DIGIKAM_VIDEO_TAGGER_DB_PORT = '3307'
$env:DIGIKAM_VIDEO_TAGGER_DB_USER = 'root'
$env:DIGIKAM_VIDEO_TAGGER_DB_PASSWORD = ''
$env:DIGIKAM_VIDEO_TAGGER_DB_NAME = 'digikam'
