from __future__ import annotations

import numpy as np
import pytest

from digikam_video_tagger.digikam_db import PersonEmbedding
from digikam_video_tagger.models import FaceTagger


class FakeDetector:
    def setInputSize(self, size: tuple[int, int]) -> None:
        self.size = size

    def detect(self, image: np.ndarray):
        face = np.array(
            [[0, 0, 10, 10, 1, 1, 2, 2, 3, 3, 4, 4, 0.91]], dtype=np.float32
        )
        return None, face


class FakeRecognizer:
    def __init__(self, feature: np.ndarray) -> None:
        self._feature = feature

    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        return image

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        return self._feature.copy()


class NoFeatureRecognizer:
    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        return image

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        raise RuntimeError("feature extraction should not run")


def make_tagger(
    feature: np.ndarray,
    gallery: list[PersonEmbedding],
    recognizer_cls: type | None = None,
) -> FaceTagger:
    tagger = object.__new__(FaceTagger)
    tagger.detector = FakeDetector()
    tagger.recognizer = (recognizer_cls or FakeRecognizer)(feature)
    tagger.gallery = gallery
    tagger.recognition_distance = 0.50
    return tagger


def test_detect_faces_returns_embedding_with_empty_gallery() -> None:
    feature = np.arange(1, 129, dtype=np.float32)
    detections = make_tagger(feature, []).detect_faces(
        np.zeros((16, 16, 3), dtype=np.uint8)
    )
    assert len(detections) == 1
    assert detections[0].name is None
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].embedding.shape == (128,)
    assert np.linalg.norm(detections[0].embedding) == pytest.approx(1.0)


def test_detect_faces_returns_known_name_and_detect_remains_compatible() -> None:
    feature = np.ones(128, dtype=np.float32)
    vector = feature / np.linalg.norm(feature)
    gallery = [PersonEmbedding(1, "Mom", vector)]
    tagger = make_tagger(feature, gallery)
    detections = tagger.detect_faces(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections[0].name == "Mom"
    assert detections[0].confidence == pytest.approx(1.0)
    assert np.linalg.norm(detections[0].embedding) == pytest.approx(1.0)
    assert tagger.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == (
        1,
        {"Mom": pytest.approx(1.0)},
    )


def test_detect_faces_with_no_faces_returns_empty_list() -> None:
    tagger = object.__new__(FaceTagger)
    tagger.recognition_distance = 0.50

    class EmptyDetector:
        def setInputSize(self, size: tuple[int, int]) -> None:
            pass

        def detect(self, image: np.ndarray):
            return None, None

    tagger.detector = EmptyDetector()
    tagger.recognizer = FakeRecognizer(np.zeros(128, dtype=np.float32))
    tagger.gallery = []
    detections = tagger.detect_faces(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections == []


def test_detect_faces_drops_zero_norm_feature() -> None:
    detections = make_tagger(
        np.zeros(128, dtype=np.float32), [PersonEmbedding(1, "Mom", np.ones(128))]
    ).detect_faces(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections == []


def test_detect_faces_drops_gallery_miss_beyond_threshold() -> None:
    feature = np.zeros(128, dtype=np.float32)
    feature[0] = 1.0
    vector = np.zeros(128, dtype=np.float32)
    vector[1] = 1.0
    gallery = [PersonEmbedding(1, "Mom", vector)]
    detections = make_tagger(feature, gallery).detect_faces(
        np.zeros((16, 16, 3), dtype=np.uint8)
    )
    assert len(detections) == 1
    assert detections[0].name is None
    assert detections[0].confidence == pytest.approx(0.91)


def test_legacy_detect_with_empty_gallery_skips_feature_extraction() -> None:
    tagger = object.__new__(FaceTagger)
    tagger.detector = FakeDetector()
    tagger.recognizer = NoFeatureRecognizer()
    tagger.gallery = []
    tagger.recognition_distance = 0.50
    face_count, people = tagger.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert face_count == 1
    assert people == {}


def test_detect_reports_full_detector_count_despite_invalid_feature() -> None:
    feature = np.zeros(128, dtype=np.float32)
    gallery = [PersonEmbedding(1, "Mom", np.ones(128))]
    face_count, people = make_tagger(feature, gallery).detect(
        np.zeros((16, 16, 3), dtype=np.uint8)
    )
    assert face_count == 1
    assert people == {}
