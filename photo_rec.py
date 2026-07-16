import cv2
import datetime
import time
from queue import Empty


class Photo_Record:
    """Persist alert frames to the configured snapshot directory."""

    def __init__(self,
                 get_frames,
                 video_event,
                 done_video_event,
                 set_dir_path,
                 _queue):
        self.get_frames = get_frames
        self.video_event = video_event
        self.done_video_event = done_video_event
        self.start_time = None
        self._queue = _queue
        self.cnt = 5

    def photo(self):
        print(f"DEBUG: attempt to launch photo rec")

        # The save directory is supplied by the settings screen through this queue.
        dir_path = self._queue.get()
        while True:
            try:
                frame = self.get_frames.get(timeout=1)

                timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
                filename = f"{dir_path}/{timestamp}.jpg"
                # Write JPEGs at maximum quality so evidence snapshots keep detail.
                cv2.imwrite(
                    filename,
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY, 100

                    ])
                print(f"Photo saved: {filename}")
                time.sleep(0.5)
            except Empty:
                continue
