from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .digikam_db import PersonEmbedding


@dataclass(frozen=True)
class OpenCVTarget:
    backend: int
    target: int
    name: str


def select_opencv_target(require_opencl: bool = True) -> OpenCVTarget:
    have_opencl = bool(cv2.ocl.haveOpenCL())
    if have_opencl:
        cv2.ocl.setUseOpenCL(True)
        return OpenCVTarget(cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL, "OpenCL")
    if require_opencl:
        raise RuntimeError("The Python OpenCV runtime cannot access OpenCL")
    return OpenCVTarget(cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU")


def _letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


class YoloObjectTagger:
    def __init__(
        self,
        model_path: Path,
        class_names_path: Path,
        target: OpenCVTarget,
        *,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
    ) -> None:
        self.class_names = [
            line.strip() for line in class_names_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(target.backend)
        self.net.setPreferableTarget(target.target)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image: np.ndarray) -> dict[str, float]:
        padded, scale, left, top = _letterbox(image, 640)
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        prediction = np.squeeze(self.net.forward())
        if prediction.ndim != 2:
            return {}
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return {}

        class_scores = prediction[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        selected = confidences >= self.confidence_threshold
        rows = prediction[selected]
        selected_ids = class_ids[selected]
        selected_confidences = confidences[selected]
        if rows.size == 0:
            return {}

        boxes: list[list[int]] = []
        for row in rows:
            center_x, center_y, width, height = row[:4]
            x = int((center_x - width / 2 - left) / scale)
            y = int((center_y - height / 2 - top) / scale)
            boxes.append([x, y, max(1, int(width / scale)), max(1, int(height / scale))])

        kept = cv2.dnn.NMSBoxes(
            boxes,
            selected_confidences.astype(float).tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )
        output: dict[str, float] = {}
        for index in np.array(kept).reshape(-1).tolist() if len(kept) else []:
            class_id = int(selected_ids[index])
            if 0 <= class_id < len(self.class_names):
                name = self.class_names[class_id]
                output[name] = max(output.get(name, 0.0), float(selected_confidences[index]))
        return output


class FaceTagger:
    def __init__(
        self,
        yunet_path: Path,
        sface_path: Path,
        target: OpenCVTarget,
        gallery: list[PersonEmbedding],
        *,
        detection_threshold: float = 0.70,
        recognition_distance: float = 0.50,
    ) -> None:
        self.detector = cv2.FaceDetectorYN_create(
            str(yunet_path),
            "",
            (112, 112),
            detection_threshold,
            0.3,
            5000,
            target.backend,
            target.target,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(sface_path), "", target.backend, target.target
        )
        self.gallery = gallery
        self.recognition_distance = recognition_distance

    def detect(self, image: np.ndarray) -> tuple[int, dict[str, float]]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return 0, {}

        people: dict[str, float] = {}
        if not self.gallery:
            return len(faces), people

        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm <= 0:
                continue
            feature /= norm
            best: PersonEmbedding | None = None
            best_distance = float("inf")
            for sample in self.gallery:
                distance = 1.0 - float(np.dot(feature, sample.vector))
                l2_distance = float(np.linalg.norm(feature - sample.vector))
                if distance < self.recognition_distance and l2_distance < 1.05 and distance < best_distance:
                    best = sample
                    best_distance = distance
            if best is not None:
                confidence = max(0.0, min(1.0, 1.0 - best_distance))
                people[best.name] = max(people.get(best.name, 0.0), confidence)
        return len(faces), people
