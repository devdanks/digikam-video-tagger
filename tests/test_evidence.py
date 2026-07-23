from digikam_video_tagger.evidence import EvidenceAccumulator


def test_evidence_requires_repeated_frames() -> None:
    evidence = EvidenceAccumulator()
    evidence.add_frame({"cat": 0.8, "person": 0.7})
    evidence.add_frame({"cat": 0.9})
    evidence.add_frame({"dog": 0.95})

    accepted = evidence.accepted(min_hits=2, min_frame_ratio=0.1)

    assert [item.label for item in accepted] == ["cat"]
    assert accepted[0].hits == 2
    assert accepted[0].max_confidence == 0.9


def test_single_frame_video_can_pass_min_hits() -> None:
    evidence = EvidenceAccumulator()
    evidence.add_frame({"car": 0.75})

    accepted = evidence.accepted(min_hits=2, min_frame_ratio=0.5)

    assert [item.label for item in accepted] == ["car"]
