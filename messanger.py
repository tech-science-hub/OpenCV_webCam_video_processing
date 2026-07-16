import requests
import datetime
import time
import cv2
import json
from pathlib import Path

DEFAULT_BOT_DATA = Path(__file__).resolve().parent / "system_settings" / "bot.json"


class Telegram:
    """Wait for detection events and send the matching frame to Telegram."""

    def __init__(self, detection_event, photo_detection_event, frame_queue, bot_data=DEFAULT_BOT_DATA):
        self.detection_event = detection_event
        self.photo_detection_event = photo_detection_event
        self.frame_queue = frame_queue
        self.bot_data = Path(bot_data)
        with self.bot_data.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

    def message_menager(self):
        message = f"pc on the line"
        while True:
            try:
                # The pipeline sets this event only after an object and motion are detected.
                self.photo_detection_event.wait()
                time_point = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
                caption = f"_{time_point}"

                self.send_photo(caption)

                self.photo_detection_event.clear()

            except Exception as e:
                print(f"error with tg server responce {e}")

    def send_photo(self, caption):
        print("send photo initialised")
        """Send a photo (JPG, PNG)"""

        bot_api = self.data.get("token")
        chatId = self.data.get("chat_id")

        url = f'https://api.telegram.org/bot{bot_api}/sendPhoto'

        # The frame is passed in memory; no temporary image file is needed for Telegram upload.
        frame = self.frame_queue.get()
        flag, buff = cv2.imencode(".jpg", frame)
        files = {'photo': buff}
        data = {'chat_id': chatId}
        if caption:
            data['caption'] = caption

        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("✅ Photo sent successfully!")
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None


