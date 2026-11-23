from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from src.counting.line_counter import LineCounter
from src.statistics.occupancy_store import OccupancyStore
from src.tracking.centroid_tracker import CentroidTracker
from src.device import get_device


class PersonDemo:
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.5,
        device_name: str = "cpu",
        max_width: int = 640,
        infer_every: int = 2,
    ):
        self.device = get_device(device_name)
        self.max_width = max_width
        self.infer_every = infer_every
        self.frame_index = 0
        self.last_detections = []
        self.last_tracked = {}
        self.previous_time = None
        self.fps = 0.0
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo exportado: {model_path}. Entrene primero."
            )
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()
        self.score_threshold = score_threshold
        self.tracker = CentroidTracker()
        self.line_counter = None
        self.store = OccupancyStore()
        self.line_points = []
        self.transform = T.Compose([T.ConvertImageDtype(torch.float32)])

    def set_line_from_default(self, frame_width, frame_height):
        if self.line_counter is None:
            start = (0, frame_height // 2)
            end = (frame_width, frame_height // 2)
            self.line_counter = LineCounter(start, end)

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.line_points.append((x, y))
            if len(self.line_points) == 2:
                self.line_counter = LineCounter(
                    self.line_points[0],
                    self.line_points[1],
                )
                self.line_points = []

    def detect(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255)
        with torch.no_grad():
            prediction = self.model(tensor.to(self.device))

        detections = []
        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= self.score_threshold:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                detections.append(np.array([center[0], center[1], x1, y1, x2, y2]))
        return detections

    def run(self, source):
        source_value = int(source) if str(source).isdigit() else source
        capture = cv2.VideoCapture(source_value)
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")

        cv2.namedWindow("Conteo de personas")
        cv2.setMouseCallback("Conteo de personas", self.mouse_callback)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            self.set_line_from_default(width, height)

            inference_frame = frame
            scale_x = scale_y = 1.0
            if width > self.max_width:
                scale_x = self.max_width / width
                scale_y = self.max_width / width
                inference_height = int(height * scale_y)
                inference_frame = cv2.resize(
                    frame,
                    (self.max_width, inference_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.frame_index % self.infer_every == 0:
                detections = self.detect(inference_frame)
                for detection in detections:
                    detection[[0, 2, 4]] *= scale_x
                    detection[[1, 3, 5]] *= scale_y
                self.last_detections = detections
                centroids = [detection[:2] for detection in detections]
                self.last_tracked = self.tracker.update(centroids)
            else:
                detections = self.last_detections

            tracked = self.last_tracked
            entries, exits, occupancy = self.line_counter.update(tracked)

            current_time = time.perf_counter()
            if self.previous_time is not None:
                elapsed = current_time - self.previous_time
                if elapsed > 0:
                    instantaneous = 1.0 / elapsed
                    self.fps = (
                        instantaneous
                        if self.fps == 0
                        else self.fps * 0.9 + instantaneous * 0.1
                    )
            self.previous_time = current_time
            self.frame_index += 1

            for detection in detections:
                x1, y1, x2, y2 = detection[2:].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for identifier, centroid in tracked.items():
                cv2.circle(frame, tuple(np.asarray(centroid, dtype=int)), 4, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"ID {identifier}",
                    tuple(np.asarray(centroid, dtype=int) + (8, -8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            cv2.line(
                frame,
                self.line_counter.start,
                self.line_counter.end,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Entradas: {entries}  Salidas: {exits}  Ocupacion: {occupancy}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}  Dispositivo: {self.device}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Conteo de personas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                self.line_counter.reset()
                self.tracker = CentroidTracker()
            if key == ord("l"):
                self.line_points = []

        self.store.record(entries, exits, occupancy)
        capture.release()
        cv2.destroyAllWindows()

