import queue
import threading
import os
from queue import Queue
from Pipeline import CameraPipeline

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



def main():

    # Queues with size 1 intentionally favor live frames over backlog processing.
    _queue = Queue(maxsize=1)
    pic_queue = Queue(maxsize=1)
    tg_queue = Queue(maxsize=5)
    stream_queue = Queue(maxsize=1)
    inference_queue = Queue(maxsize=1)
    load_data_signal = threading.Event()
    start_event = threading.Event()
    start_stream = threading.Event()
    start_recognition = threading.Event()
    photo_detection_event = threading.Event()

    cp = CameraPipeline(
        start_stream,
        start_recognition,
        inference_queue,
        photo_detection_event,
        pic_queue,
        tg_queue,
        start_event,
        stream_queue,
        load_data_signal,
        _queue)

    try:
        cp.start()
        cp.run()
    finally:
        cp.stop()

if __name__ == "__main__":
    main()



