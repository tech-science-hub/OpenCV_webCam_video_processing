import queue
import cv2
import time
import threading
import cvzone as cvz
from queue import Queue, Full
from ultralytics import YOLO

time_out = 15

row_size = 100
left_margin = 10
text_color = (0, 0, 255)
font_size = 2
font_thickness = 1
_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)


class Camera:
    def __init__(self,
                 cameraID,
                 url,

                 queue_to_detection,
                 restart):

        self.cameraID           = cameraID
        self.url                = url
        self.start_time         = time.time()
        self.frame_data         = Queue(maxsize=1)
        self.person_detected    = False
        self.fails              = 0
        self.cap                = None
        self._is_running        = True
        self.restart_event      = restart
        self.thread             = None
        self.join               = None
        self.frame_lock         = threading.Lock()
        self.last_frame         = None
        self.start_time         = time.time()
        self._is_connected      = None
        self.export_frame       = None
        self.model              = YOLO('Ip_cam_survilance/yolo11n.pt')
        self.counter            = 0
        self.fps                = 0
        self.queue_to_detection = queue_to_detection
        self.fps_avg_frame_count = 15
        self.last_alert_time    = 0
        print(id(self))
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

        while self.cap.isOpened():

            is_connected = True
            success, frame = self.cap.read()

            if not success:
                print("no frames")
                _is_connected = False
            results = self.model.predict(frame, stream=True, conf=0.7, verbose=False)

            self.counter += 1

            for result in results:
                bbox = result.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cls = int(box.cls[0])
                    current_cls = self.model.names[cls]
                    conf = round(box.conf[0].item(), 2)

                    text_location = (_MARGIN + x1, _MARGIN + _ROW_SIZE + y1)

                    if current_cls == 'person' or current_cls == 'car':

                        if conf > 0.75:
                            person_detected = True

                            cvz.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), 15, 2, 1, (0, 255, 0), (0, 0, 255))

                            cvz.putTextRect(frame, f"{self.model.names[cls]} {conf}", (max(0, x1), max(40, y1)), scale=0.6,
                                            offset=3, thickness=1)

                    else:
                        person_detected = False

            if self.counter % self.fps_avg_frame_count == 0:
                end_time = time.time()
                self.fps = self.fps_avg_frame_count / (end_time - self.start_time)
                self.start_time = time.time()

            fps_text = 'FPS = {:.1f}'.format(self.fps)
            text_location = (left_margin, row_size)
            cvz.putTextRect(frame,
                            fps_text,
                            text_location,
                            1, 1,
                            (255, 255, 255),
                            (255, 0, 255)
                            )

            success, buffer = cv2.imencode(".jpg", frame.copy())
            now = time.time()
            if now - self.last_alert_time > 2:
                self.last_alert_time = now

                if success:
                    if self.frame_data.full():
                        try:
                            self.frame_data.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self.frame_data.put_nowait(buffer.tobytes())
                    except queue.Full:
                        pass

            frame = cv2.resize(frame, (960, 480))
            cv2.imshow(f'camera {self.cameraID}', frame)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()


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
