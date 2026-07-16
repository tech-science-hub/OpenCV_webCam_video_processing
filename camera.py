import queue
import cv2
import time
import threading
import cvzone as cvz
from datetime import datetime

time_out = 15
fps_avg_frame_count = 15
row_size = 20
left_margin = 5
text_color = (0, 0, 255)
font_size = 2
font_thickness = 1
_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)


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
        self.start_time = time.time()
        self._is_connected = None
        self.export_frame = None
        self.counter = 0
        self.fps = 0
        self.queue_to_detection = queue_to_detection
        self.queue_to_stream = queue_to_stream
        self.type = None
        self.previous_frame = None

    def start(self):
        self._is_running = True
        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        print(f"camera {self.cameraID} runs")
        self.thread.start()

    def run(self):
        self.cap = cv2.VideoCapture(self.url)

        while self._is_running:
            # The dashboard controls this event; cameras wait here until streaming is enabled.
            self.start_stream.wait()
            success, image = self.cap.read()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            if not success:
                time.sleep(1)
                continue
            h, w = image.shape[:2]

            if self.counter % fps_avg_frame_count == 0:
                end_time = time.time()
                self.fps = fps_avg_frame_count / (end_time - self.start_time)
                self.start_time = time.time()

            fps_text = 'FPS = {:.1f}'.format(self.fps)
            text_location = (left_margin, row_size)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            (text_w, text_h), _ = cv2.getTextSize(
                timestamp,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cvz.putTextRect(image,
                            fps_text,
                            text_location,
                            1, 1,
                            (255, 255, 255),
                            (0, 0, 0)
                            )
            cvz.putTextRect(image,
                            timestamp,
                            (w-text_w, text_h),
                            1, 1,
                            (255, 255, 255),
                            (0, 0, 0)
                            )
            with self.frame_lock:
                self.last_frame = image.copy()

            payload = (
                self.cameraID,
                time.time(),
                image
            )

            # Keep only the newest frame so detection and streaming do not lag behind live video.
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

    def restart(self):
        print(f"restart camera {self.cameraID}")
        self.stop()
        time.sleep(5)
        self.launch()

    def status(self):
        if self.thread.is_alive():
            return False
        else:
            return True

    def stop(self):
        self._is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
