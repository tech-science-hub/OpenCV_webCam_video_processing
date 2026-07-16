import json
from pathlib import Path


DEFAULT_CAMERAS = {"cameras": []}
DEFAULT_BOT = {
    "token": "",
    "api_url": "",
    "username": "",
    "chat_id": "",
}
DEFAULT_SETTINGS = {
    "schedule_time": "",
    "photo_save_dir": "",
    "cleanup_days": "",
    "model_path": "",
    "inference_every_n_frames": 3,
    "stream_max_fps": 8,
}


class ConfigStore:
    """Read and write dashboard-managed JSON configuration files."""

    def __init__(self, settings_dir):
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(exist_ok=True)
        self.cameras_path = self.settings_dir / "cameras.json"
        self.bot_path = self.settings_dir / "bot.json"
        self.settings_path = self.settings_dir / "settings.json"
        self.ensure_files()

    def ensure_files(self):
        self._ensure_json(self.cameras_path, DEFAULT_CAMERAS)
        self._ensure_json(self.bot_path, DEFAULT_BOT)
        self._ensure_json(self.settings_path, DEFAULT_SETTINGS)

    def read_cameras(self):
        data = self._read_json(self.cameras_path, DEFAULT_CAMERAS)
        cameras = data.get("cameras", [])
        data["configured"] = any(str(cam.get("source", "")).strip() != "" for cam in cameras)
        return data

    def save_cameras(self, data):
        self._write_json(self.cameras_path, data or DEFAULT_CAMERAS)

    def read_bot(self):
        data = self._read_json(self.bot_path, DEFAULT_BOT)
        data["configured"] = self.is_bot_configured(data)
        return data

    def save_bot(self, data):
        config = {
            "token": data.get("token", ""),
            "api_url": data.get("api_url", ""),
            "username": data.get("username", ""),
            "chat_id": data.get("chat_id", ""),
        }
        self._write_json(self.bot_path, config)

    def read_settings(self):
        return self._read_json(self.settings_path, DEFAULT_SETTINGS)

    def save_settings(self, data):
        self._write_json(self.settings_path, data or DEFAULT_SETTINGS)

    @staticmethod
    def is_bot_configured(data):
        return str(data.get("token", "")).strip() != "" and str(data.get("chat_id", "")).strip() != ""

    @staticmethod
    def _ensure_json(path, default):
        if not path.exists():
            ConfigStore._write_json(path, default)

    @staticmethod
    def _read_json(path, default):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            ConfigStore._write_json(path, default)
            return default.copy()

    @staticmethod
    def _write_json(path, data):
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
