import queue

import time
import threading
import torch
from pathlib import Path
from ultralytics import YOLO


class ProcessFrame:
    """Run YOLO detection in a background thread and cache recent results by camera."""

    def __init__(self, queue_to_detection, start_event, model_path=None):
        self.queue_to_detection = queue_to_detection
        self.start_event = start_event
        self._isRunning = False
        self.thread = None
        self.result_lock = threading.Lock()
        self.model_path = model_path or Path(__file__).resolve().parent / "Ip_cam_survilance" / "yolo11n.pt"
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()
        self.model = YOLO(str(self.model_path))
        self.output_results = {}

    def start(self):
        self._isRunning = True
        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        print("recognition is started")
        self.thread.start()

    def run(self):
        while self._isRunning:
            if not self.start_event.is_set():
                time.sleep(0.1)
                continue
            try:

                camera_id, time_id, image = self.queue_to_detection.get(timeout=1)
                # Drop stale queued frames; the pipeline only needs the latest detection result.
                if self.queue_to_detection.full():
                    self.queue_to_detection.get_nowait()


            except queue.Empty:
                continue

            detections = []

            results = self.model.predict(
                                        image,
                                        conf=0.5,
                                        verbose=False,
                                        device=self.device,
                                        half=self.use_half,
                                        imgsz=320,
                                        stream=True
                                        )

            for result in results:
                bbox = result.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    current_cls = self.model.names[cls]
                    conf = round(box.conf[0].item(), 2)
                    detections.append(
                                {
                                    "class": current_cls,
                                    "conf": conf,
                                    "bbox": (x1, y1, x2, y2)

                }
                    )
                with self.result_lock:
                    self.output_results[camera_id] = {
                        "time": time.time(),
                        "timeID": time_id,
                        "detections": detections
                    }

    def get_detections(self, camera_id):
        with self.result_lock:

            result = self.output_results.get(camera_id)
            if result is None:
                return []
            # Ignore old detections so boxes are not drawn on newer frames.
            if time.time() - result["timeID"] > 0.5:
                return []

            return result["detections"]

    def stop(self):
        self._isRunning = False

        if self.thread and self.thread.is_alive():
            self.thread.join()
