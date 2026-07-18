import datetime
import json
from pathlib import Path
from queue import Empty

import cv2
import requests

DEFAULT_BOT_DATA = Path(__file__).resolve().parent / "system_settings" / "bot.json"


class Telegram:
    """Wait for detection events and send the matching frame to Telegram."""

    def __init__(self, detection_event, photo_detection_event, frame_queue, bot_data=DEFAULT_BOT_DATA):
        self.detection_event = detection_event
        self.photo_detection_event = photo_detection_event
        self.frame_queue = frame_queue
        self.bot_data = Path(bot_data)
        self.data = self._load_bot_data()

    def message_menager(self):
        while True:
            try:
                self.photo_detection_event.wait()
                while True:
                    try:
                        frame = self.frame_queue.get(timeout=1)
                    except Empty:
                        self.photo_detection_event.clear()
                        break
                    time_point = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
                    caption = f"_{time_point}"
                    self.send_photo(frame, caption)
            except Exception as e:
                print(f"error with tg server responce {e}")

    def send_photo(self, frame, caption):
        print("send photo initialised")
        self.data = self._load_bot_data()
        bot_api = self.data.get("token")
        chatId = self.data.get("chat_id")
        if not bot_api or not chatId:
            print("Telegram bot token or chat_id is missing.", flush=True)
            return None

        url = f"https://api.telegram.org/bot{bot_api}/sendPhoto"
        flag, buff = cv2.imencode(".jpg", frame)
        if not flag:
            print("Failed to encode frame for Telegram.", flush=True)
            return None

        files = {"photo": (f"{caption or 'snapshot'}.jpg", buff.tobytes(), "image/jpeg")}
        data = {"chat_id": chatId}
        if caption:
            data["caption"] = caption

        response = requests.post(url, data=data, files=files, timeout=15)
        if response.status_code == 200:
            print("✅ Photo sent successfully!")
            return response.json()

        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

    def _load_bot_data(self):
        try:
            with self.bot_data.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            print(f"Failed to load Telegram bot config: {exc}", flush=True)
            return {}
