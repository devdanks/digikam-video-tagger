from digikam_video_tagger.config import DEFAULT_STAGING_DIR, LOCAL_APP_DATA


def test_default_staging_directory_is_managed_by_the_application() -> None:
    assert DEFAULT_STAGING_DIR == LOCAL_APP_DATA / "digikam-video-tagger" / "staging"
