import glob
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import cvzone as cvz

fps_avg_frame_count = 15
row_size = 20
left_margin = 5


class Camera:
    """Capture one camera stream and publish each fresh frame to worker queues."""

    def __init__(self, cameraID, url, frame_data, queue_to_detection, queue_to_stream, restart, start_stream):
        self.start_stream = start_stream
        self.cameraID = cameraID
        self.url = url
        self.start_time = time.time()
        self.frame_data = frame_data
        self.person_detected = False
        self.fails = 0
        self.cap = None
        self._is_running = False
        self.restart_event = restart
        self.thread = None
        self.join = None
        self.frame_lock = threading.Lock()
        self.last_frame = None
        self._is_connected = None
        self.export_frame = None
        self.counter = 0
        self.fps = 0
        self.queue_to_detection = queue_to_detection
        self.queue_to_stream = queue_to_stream
        self.type = None
        self.previous_frame = None
        self.read_failures = 0

    def start(self):
        self._is_running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        print(f"camera {self.cameraID} runs")
        self.thread.start()

    def run(self):
        if isinstance(self.url, int):
            video_nodes = sorted(glob.glob("/dev/video*"))
            if not video_nodes:
                print(
                    f"Camera {self.cameraID}: local camera index {self.url} requested, "
                    "but no /dev/video* devices were found. Check USB connection, power, "
                    "kernel driver support, and whether the webcam is detected by Linux.",
                    flush=True,
                )
                self._is_running = False
                return

            expected_node = f"/dev/video{self.url}"
            if not os.path.exists(expected_node):
                print(
                    f"Camera {self.cameraID}: requested local camera index {self.url}, "
                    f"but {expected_node} does not exist. Available devices: {', '.join(video_nodes)}",
                    flush=True,
                )
                self._is_running = False
                return

        if not self._open_capture():
            self._is_running = False
            return

        while self._is_running:
            self.start_stream.wait()
            success, image = self.cap.read()

            if not success:
                self.read_failures += 1
                print(
                    f"Camera {self.cameraID}: frame read failed from source {self.url}. Retrying...",
                    flush=True,
                )
                if self.read_failures >= 5:
                    print(
                        f"Camera {self.cameraID}: reopening source after {self.read_failures} consecutive read failures.",
                        flush=True,
                    )
                    self._reopen_capture()
                time.sleep(1)
                continue

            self.read_failures = 0
            h, w = image.shape[:2]

            if self.counter % fps_avg_frame_count == 0:
                end_time = time.time()
                self.fps = fps_avg_frame_count / (end_time - self.start_time)
                self.start_time = time.time()

            fps_text = "FPS = {:.1f}".format(self.fps)
            text_location = (left_margin, row_size)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            (text_w, text_h), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cvz.putTextRect(image, fps_text, text_location, 1, 1, (255, 255, 255), (0, 0, 0))
            cvz.putTextRect(image, timestamp, (w - text_w, text_h), 1, 1, (255, 255, 255), (0, 0, 0))

            with self.frame_lock:
                self.last_frame = image.copy()

            payload = (self.cameraID, time.time(), image)

            for q in (self.queue_to_detection, self.queue_to_stream):
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

            self.counter += 1

    def get_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame

    def stop(self):
        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            return True

        source = f"camera index {self.url}" if isinstance(self.url, int) else str(self.url)
        print(
            f"Camera {self.cameraID}: failed to open {source}. "
            "If this is a USB webcam, verify /dev/video* exists and the user has access. "
            "If this is an RTSP stream, verify the URL and network reachability.",
            flush=True,
        )
        return False

    def _reopen_capture(self):
        if self.cap:
            self.cap.release()
        time.sleep(2)
        if self._open_capture():
            self.read_failures = 0
            print(f"Camera {self.cameraID}: source reopened successfully.", flush=True)
