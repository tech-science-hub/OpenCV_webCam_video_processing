from datetime import datetime, timedelta
import os
import threading
import time
import queue


class SurveillanceScheduler:
    """Own scheduled start and snapshot cleanup background jobs."""

    def __init__(self, start_stream, start_recognition, photo_dir_queue):
        self.start_stream = start_stream
        self.start_recognition = start_recognition
        self.photo_dir_queue = photo_dir_queue
        self.schedule_time = None
        self.photo_dir_path = None
        self.cleanup_days = None
        self.cleanup_enabled = False
        self.schedule_enabled = False
        self.cleanup_thread = None
        self.autostart_thread = None

    def apply_settings(self, settings):
        self.schedule_time = settings.get("schedule_time")
        self.photo_dir_path = settings.get("photo_save_dir")
        self.cleanup_days = settings.get("cleanup_days")
        self._publish_photo_dir()
        self._start_cleanup_if_needed()
        self._start_autostart_if_needed()

    def status(self):
        return {
            "schedule_time": self.schedule_time,
            "cleanup_days": self.cleanup_days,
            "photo_path_set": bool(self.photo_dir_path),
        }

    def _publish_photo_dir(self):
        if not self.photo_dir_path:
            return
        try:
            self.photo_dir_queue.put_nowait(self.photo_dir_path)
        except queue.Full:
            pass

    def _start_cleanup_if_needed(self):
        self.cleanup_enabled = self.cleanup_days is not None and str(self.cleanup_days).strip() != ""
        if not self.cleanup_enabled:
            return
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(
                target=self.delete_old_pictures,
                daemon=True,
            )
            self.cleanup_thread.start()

    def _start_autostart_if_needed(self):
        self.schedule_enabled = self.schedule_time is not None and str(self.schedule_time).strip() != ""
        if not self.schedule_enabled:
            return
        if self.autostart_thread is None or not self.autostart_thread.is_alive():
            self.autostart_thread = threading.Thread(
                target=self.set_auto_start,
                daemon=True,
            )
            self.autostart_thread.start()

    def delete_old_pictures(self):
        while self.cleanup_enabled:
            try:
                days = int(self.cleanup_days)
            except (TypeError, ValueError):
                self.cleanup_enabled = False
                break

            if days <= 0:
                self.cleanup_enabled = False
                break

            if not self.photo_dir_path or not os.path.isdir(self.photo_dir_path):
                time.sleep(60)
                continue

            now = time.time()
            max_age = days * 24 * 60 * 60
            for filename in os.listdir(self.photo_dir_path):
                path = os.path.join(self.photo_dir_path, filename)
                if not os.path.isfile(path):
                    continue
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if now - os.path.getmtime(path) > max_age:
                    os.remove(path)

            time.sleep(3600)

    def set_auto_start(self):
        while self.schedule_enabled:
            target = self.calculate_next_start()
            wait = max(0, (target - datetime.now()).total_seconds())
            time.sleep(wait)

            if not self.schedule_enabled:
                continue

            self.start_stream.set()
            time.sleep(5)
            self.start_recognition.set()

    def calculate_next_start(self):
        hour, minute = map(int, self.schedule_time.split(":"))
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target
