import json
import os
import queue
import threading
import time
from pathlib import Path

import torch
from ultralytics import YOLO


class ProcessFrame:
    """Run YOLO detection in a background thread and cache recent results by camera."""

    def __init__(self, queue_to_detection, start_event, model_path=None, settings_path=None):
        self.queue_to_detection = queue_to_detection
        self.start_event = start_event
        self._isRunning = False
        self.thread = None
        self.result_lock = threading.Lock()
        self.project_root = Path(__file__).resolve().parent
        self.settings_path = Path(settings_path) if settings_path else self.project_root / "system_settings" / "settings.json"
        self.settings = self._read_settings()
        self.default_model_path = self.project_root / "Ip_cam_survilance" / "yolo11n.pt"
        self.model_path = self._resolve_model_path(model_path)
        self.is_tensorrt_engine = self.model_path.suffix.lower() == ".engine"
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available() and not self.is_tensorrt_engine
        self.model = YOLO(str(self.model_path))
        self.output_results = {}
        self.frame_skip = self._parse_positive_int(self.settings.get("inference_every_n_frames"), default=1)
        self.frame_counters = {}

    def _resolve_model_path(self, explicit_model_path):
        candidate = explicit_model_path or os.environ.get("SURVEILLANCE_MODEL_PATH")
        if not candidate:
            candidate = str(self.settings.get("model_path", "")).strip()
        if not candidate:
            return self.default_model_path

        resolved = Path(candidate).expanduser()
        if not resolved.is_absolute():
            resolved = (self.project_root / resolved).resolve()
        if resolved.exists():
            return resolved

        print(
            f"Configured model path '{resolved}' was not found. Falling back to '{self.default_model_path}'.",
            flush=True,
        )
        return self.default_model_path

    def _read_settings(self):
        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _parse_positive_int(value, default):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def start(self):
        self._isRunning = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        print("recognition is started")
        self.thread.start()

    def run(self):
        while self._isRunning:
            if not self.start_event.is_set():
                time.sleep(0.1)
                continue
            try:
                camera_id, time_id, image = self.queue_to_detection.get(timeout=1)
                if self.queue_to_detection.full():
                    self.queue_to_detection.get_nowait()
            except queue.Empty:
                continue

            self.frame_counters[camera_id] = self.frame_counters.get(camera_id, 0) + 1
            if self.frame_counters[camera_id] % self.frame_skip != 0:
                continue

            detections = []
            predict_kwargs = {
                "conf": 0.5,
                "verbose": False,
                "imgsz": 320,
                "stream": True,
            }
            if not self.is_tensorrt_engine:
                predict_kwargs["device"] = self.device
                predict_kwargs["half"] = self.use_half

            results = self.model.predict(image, **predict_kwargs)

            for result in results:
                bbox = result.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    current_cls = self.model.names[cls]
                    conf = round(box.conf[0].item(), 2)
                    detections.append({"class": current_cls, "conf": conf, "bbox": (x1, y1, x2, y2)})
                with self.result_lock:
                    self.output_results[camera_id] = {
                        "time": time.time(),
                        "timeID": time_id,
                        "detections": detections,
                    }

    def get_detections(self, camera_id):
        with self.result_lock:
            result = self.output_results.get(camera_id)
            if result is None:
                return []
            if time.time() - result["timeID"] > 0.5:
                return []
            return result["detections"]

    def stop(self):
        self._isRunning = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
