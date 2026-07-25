from digikam_video_tagger.ffmpeg import FFmpegSampler


def test_sampling_filter_always_selects_first_frame() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1920, cuda=False)

    assert "isnan(prev_selected_t)" in value
    assert "gte(t-prev_selected_t\\,5)" in value
    assert "fps=" not in value
    assert "scale=1920:1920" in value
    assert "force_original_aspect_ratio=decrease" in value
    assert "force_divisible_by=2" in value
    assert "reset_sar=1" in value


def test_cuda_sampling_filter_keeps_cuda_operations_adjacent() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1920, cuda=True)

    assert "select=" in value
    assert value.index("hwupload") < value.index("scale_cuda")
    assert value.index("scale_cuda") < value.index("hwdownload")
    assert "isnan(prev_selected_t)" in value


def test_cpu_sampling_filter_constrains_both_dimensions() -> None:
    value = FFmpegSampler.sampling_filter(5.0, 1280, cuda=False)

    assert "scale=1280:1280" in value
    assert "force_original_aspect_ratio=decrease" in value
    assert "force_divisible_by=2" in value
