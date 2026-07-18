import glob
import platform
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
        if not self._open_capture():
            self._is_running = False
            return

        while self._is_running:
            self.start_stream.wait()
            if not self._is_running:
                break
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
        self.start_stream.set()
        if self.cap:
            self.cap.release()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            if "cvDestroyAllWindows" not in str(exc):
                raise

    def _open_capture(self):
        source = self._normalize_source(self.url)
        for backend in self._capture_backends(source):
            self.cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
            if self.cap.isOpened():
                backend_name = self._backend_name(backend)
                print(f"Camera {self.cameraID}: opened {source!r} using {backend_name}.", flush=True)
                return True
            self.cap.release()

        if self._is_linux_camera_index(source):
            video_nodes = sorted(glob.glob("/dev/video*"))
            available = ", ".join(video_nodes) if video_nodes else "none"
            print(
                f"Camera {self.cameraID}: Linux camera index {source} could not be opened. "
                f"Available /dev/video* devices: {available}.",
                flush=True,
            )
        else:
            print(
                f"Camera {self.cameraID}: failed to open {source!r} on {platform.system()}. "
                "For USB cameras use a numeric index such as 0. For RTSP, verify the URL and network reachability.",
                flush=True,
            )
        return False

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())
        return source

    @staticmethod
    def _capture_backends(source):
        if not isinstance(source, int):
            return [cv2.CAP_FFMPEG, cv2.CAP_ANY]

        system = platform.system().lower()
        if system == "windows":
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        if system == "linux":
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        if system == "darwin":
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    @staticmethod
    def _is_linux_camera_index(source):
        return isinstance(source, int) and platform.system().lower() == "linux"

    @staticmethod
    def _backend_name(backend):
        names = {
            cv2.CAP_ANY: "CAP_ANY",
            cv2.CAP_FFMPEG: "CAP_FFMPEG",
            cv2.CAP_DSHOW: "CAP_DSHOW",
            cv2.CAP_MSMF: "CAP_MSMF",
            cv2.CAP_V4L2: "CAP_V4L2",
        }
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            names[cv2.CAP_AVFOUNDATION] = "CAP_AVFOUNDATION"
        if backend is None:
            return "default backend"
        return names.get(backend, str(backend))

    def status(self):
        if self.thread and self.thread.is_alive():
            return False
        return True

    def _reopen_capture(self):
        if self.cap:
            self.cap.release()
        time.sleep(2)
        if self._open_capture():
            self.read_failures = 0
            print(f"Camera {self.cameraID}: source reopened successfully.", flush=True)
