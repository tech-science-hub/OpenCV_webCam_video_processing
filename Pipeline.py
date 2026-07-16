import json
import queue
import threading
import time
from pathlib import Path

import cv2
import cvzone as cvz

from Server import Server
from camera import Camera
from messanger import Telegram as TG
from photo_rec import Photo_Record
from recognition import ProcessFrame

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHOTO_DIR = PROJECT_ROOT / "photos"


class CameraPipeline:
    """Wire together capture, detection, streaming, snapshot, and Telegram workers."""

    def __init__(
        self,
        start_stream,
        start_recognition,
        inference_queue,
        photo_detection_event,
        pic_queue,
        tg_queue,
        start_event,
        stream_queue,
        load_data_signal,
        _queue,
        restart_requested,
        shutdown_requested,
    ):
        self.start_stream = start_stream
        self.start_recognition = start_recognition
        self.inference_queue = inference_queue
        self.detection_worker = None
        self.photo_detection_event = photo_detection_event
        self.pic_queue = pic_queue
        self.tg_queue = tg_queue
        self.start_event = start_event
        self.stream_queue = stream_queue
        self.cameras = {}
        self.threads = []
        self.last_alert_time = 0
        self.load_data_signal = load_data_signal
        self.restart_requested = restart_requested
        self.shutdown_requested = shutdown_requested
        self.video_event = threading.Event()
        self.done_video_event = threading.Event()
        self._queue = _queue
        self.route = self.init_route()
        self.previous_frame = {}
        self.server = None
        self.photo_record = None
        self.tg = None

    def init_route(self):
        settings_dir = PROJECT_ROOT / "system_settings"
        settings_dir.mkdir(exist_ok=True)
        return settings_dir

    def build_cameras(self):
        cam = self.route / "cameras.json"
        with cam.open("r", encoding="utf-8") as f:
            camera = json.load(f)
            for cam_data in camera["cameras"]:
                cam_id = cam_data["id"]
                self.cameras[cam_id] = Camera(
                    cameraID=cam_id,
                    url=cam_data["source"],
                    frame_data=self.pic_queue,
                    queue_to_detection=self.inference_queue,
                    queue_to_stream=self.stream_queue,
                    restart=None,
                    start_stream=self.start_stream,
                )

    def start(self):
        self.server = Server(
            start_event=self.start_event,
            stream_flow=self.stream_queue,
            start_stream=self.start_stream,
            start_recognition=self.start_recognition,
            photo_detection_event=self.photo_detection_event,
            load_data_signal=self.load_data_signal,
            _queue=self._queue,
            route=self.init_route(),
            restart_requested=self.restart_requested,
            shutdown_requested=self.shutdown_requested,
        )

        self.detection_worker = ProcessFrame(
            queue_to_detection=self.inference_queue,
            start_event=self.start_event,
            settings_path=self.route / "settings.json",
        )

        self.photo_record = Photo_Record(
            get_frames=self.pic_queue,
            video_event=self.video_event,
            done_video_event=self.done_video_event,
            set_dir_path=str(DEFAULT_PHOTO_DIR),
            _queue=self._queue,
        )

        self.tg = TG(
            detection_event=self.start_event,
            photo_detection_event=self.photo_detection_event,
            frame_queue=self.tg_queue,
            bot_data=self.route / "bot.json",
        )

        self.threads = [
            threading.Thread(target=self.server.run, daemon=True),
            threading.Thread(target=self.photo_record.photo, daemon=True),
            threading.Thread(target=self.tg.message_menager, daemon=True),
        ]

        for t in self.threads:
            t.start()

        self.detection_worker.start()
        self.build_cameras()

        for cam_id, cam in self.cameras.items():
            print(f"Starting camera {cam_id}")
            cam.start()

    def stop(self):
        if self.server:
            self.server.stop()
        if self.detection_worker:
            self.detection_worker.stop()
        for cam in self.cameras.values():
            cam.stop()

    def run(self):
        while True:
            try:
                if self.restart_requested.is_set() or self.shutdown_requested.is_set():
                    return

                self.start_recognition.wait(timeout=1)

                if not self.start_recognition.is_set():
                    continue

                camera_id, time_id, frame = self.inference_queue.get(timeout=1)

                if frame is None:
                    raise RuntimeError(f"Empty frame from camera {camera_id}")

                detections = self.detection_worker.get_detections(camera_id)

                if detections is None:
                    continue

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                if camera_id not in self.previous_frame:
                    self.previous_frame[camera_id] = gray.copy()
                    continue

                diff = cv2.absdiff(self.previous_frame[camera_id], gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)
                changed_pixels = cv2.countNonZero(thresh)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                change_ratio = changed_pixels / total_pixels
                alert_needed = False

                for detec in detections:
                    x1, y1, x2, y2 = detec["bbox"]
                    label = f'{detec["class"]} {detec["conf"]:.2f}'

                    if detec["class"] in ("person", "car"):
                        cvz.cornerRect(
                            frame,
                            (x1, y1, x2 - x1, y2 - y1),
                            30,
                            1,
                            1,
                            (0, 255, 0),
                            (0, 0, 255),
                        )

                        cvz.putTextRect(
                            frame,
                            label,
                            (max(0, x1), max(40, y1)),
                            scale=0.6,
                            offset=3,
                            thickness=1,
                        )

                        if change_ratio > 0.02:
                            alert_needed = True

                if alert_needed:
                    self.last_alert_time = time.time()

                    for target_queue in (self.pic_queue, self.tg_queue):
                        if target_queue.full():
                            try:
                                target_queue.get_nowait()
                            except queue.Empty:
                                pass

                    try:
                        self.pic_queue.put_nowait(frame)
                    except queue.Full:
                        pass

                    try:
                        self.tg_queue.put_nowait(frame)
                        self.photo_detection_event.set()
                    except queue.Full:
                        pass

                self.previous_frame[camera_id] = gray.copy()

            except queue.Empty:
                raise RuntimeError("No frames received from camera queue")
            except Exception as e:
                print(f"Runtime error inside app_run: {e}", flush=True)
                break
