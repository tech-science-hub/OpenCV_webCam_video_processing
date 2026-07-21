import os
import queue
import socket
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session, url_for
from werkzeug.serving import make_server

from auth_store import AuthStore
from config_store import ConfigStore
from scheduler import SurveillanceScheduler


class Server:
    """Flask dashboard, configuration API, and MJPEG stream server."""

    def __init__(
        self,
        start_event,
        start_stream,
        start_recognition,
        stream_flow,
        photo_detection_event,
        load_data_signal,
        _queue,
        route,
        restart_requested,
        shutdown_requested,
    ):
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get("SURVEILLANCE_SECRET_KEY", "super-secret")
        self.start_event = start_event
        self.start_stream = start_stream
        self.start_recognition = start_recognition
        self.stream_flow = stream_flow
        self.photo_detection_event = photo_detection_event
        self.load_data_signal = load_data_signal
        self.restart_requested = restart_requested
        self.shutdown_requested = shutdown_requested
        self.config = ConfigStore(route)
        self.auth = AuthStore()
        self.scheduler = SurveillanceScheduler(
            start_stream=self.start_stream,
            start_recognition=self.start_recognition,
            photo_dir_queue=_queue,
        )
        self.scheduler.apply_settings(self.config.read_settings())
        self.stream_max_fps = self._read_stream_max_fps()
        self.cam_stream = {}
        self.camera_count = len(self.config.read_cameras().get("cameras", []))
        self.bot_configured = self.config.read_bot().get("configured", False)
        self.detection_started = False
        self.connection_started = False
        self.restart_in_progress = False
        self.shutdown_in_progress = False
        self.http_server = None
        self.register_routes()
        self.stream_worker = threading.Thread(target=self.frame_reciever, daemon=True)
        self.stream_worker.start()

    def register_routes(self):
        routes = [
            ("/", self.home, None),
            ("/login", self.login, ["POST"]),
            ("/change-credentials", self.change_credentials, ["POST"]),
            ("/auth", self.get_auth, ["GET"]),
            ("/api/cameras/save", self.save_cameras, ["POST"]),
            ("/api/cameras", self.get_cameras, ["GET"]),
            ("/api/camerasStatus", self.cameraBtnStatus, ["GET"]),
            ("/api/bot/save", self.save_bot, ["POST"]),
            ("/api/bot", self.get_bot, ["GET"]),
            ("/api/botBtnStatus", self.botBtnStatus, ["GET"]),
            ("/stream/<int:camera_id>", self.stream_camera, None),
            ("/api/detection/start", self.start_detection, ["POST"]),
            ("/api/detection/stop", self.stop_detection, ["POST"]),
            ("/api/detection/connect", self.start_connection, ["POST"]),
            ("/api/detection/reset", self.reset, ["POST"]),
            ("/api/app/shutdown", self.shutdown_application, ["POST"]),
            ("/api/app/restart", self.restart_application, ["POST"]),
            ("/api/snapshots", self.snapshot, ["GET"]),
            ("/api/colorCameraBtn", self.cameraBtnStatus, ["GET"]),
            ("/api/saveGenSettings", self.saveGenSettings, ["POST"]),
            ("/api/getSettings", self.getGenSettings, ["GET"]),
            ("/api/getInfo", self.get_info, ["GET"]),
        ]

        for index, (rule, view_func, methods) in enumerate(routes):
            kwargs = {"view_func": view_func}
            if methods:
                kwargs["methods"] = methods
            endpoint = f"{view_func.__name__}_{index}"
            self.app.add_url_rule(rule, endpoint=endpoint, **kwargs)

        self.app.add_url_rule("/photo/<path:filename>", view_func=self.serve_photo, methods=["GET"], endpoint="photo")

    def run(self):
        host = os.environ.get("SURVEILLANCE_HOST", "0.0.0.0")
        port = int(os.environ.get("SURVEILLANCE_PORT", "8080"))
        print("IP Address:", socket.gethostbyname(socket.gethostname()))
        print(f"Listening on {host}:{port}")
        print(f"Open locally: http://127.0.0.1:{port}")
        self.http_server = make_server(host, port, self.app, threaded=True)
        self.http_server.serve_forever()

    def home(self):
        return render_template("surveillance_dashboard.html")

    def get_info(self):
        status = self.scheduler.status()
        status.update(
            {
                "camera_count": self.camera_count,
                "bot_set": self.bot_configured,
                "detection": self.detection_started,
                "connection": self.connection_started,
                "stream_max_fps": self.stream_max_fps,
            }
        )
        return jsonify(status)

    def stream_camera(self, camera_id):
        return Response(self.generate_stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

    def frame_reciever(self):
        while True:
            try:
                camera_id, time_id, frame = self.stream_flow.get()
                self.cam_stream[camera_id] = frame
                self.start_event.set()
            except queue.Empty:
                continue

    def get_auth(self):
        return jsonify({"username": self.auth.get_first_username()})

    def save_cameras(self):
        data = request.get_json() or {"cameras": []}
        self.config.save_cameras(data)
        self.camera_count = len(data.get("cameras", []))
        return jsonify({"success": True})

    def get_cameras(self):
        data = self.config.read_cameras()
        self.camera_count = len(data.get("cameras", []))
        return jsonify(data)

    def save_bot(self):
        data = request.get_json() or {}
        self.config.save_bot(data)
        self.bot_configured = self.config.read_bot().get("configured", False)
        return jsonify({"success": True})

    def get_bot(self):
        data = self.config.read_bot()
        self.bot_configured = data.get("configured", False)
        return jsonify(data)

    def login(self):
        data = request.get_json() or {}
        user_id = self.auth.authenticate(data.get("username"), data.get("password"))
        if user_id is None:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401
        session["user_id"] = user_id
        return jsonify({"success": True})

    def change_credentials(self):
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        data = request.get_json() or {}
        success, message = self.auth.change_credentials(
            user_id=session["user_id"],
            current_password=data.get("currentPassword"),
            new_login=data.get("newLogin"),
            new_password=data.get("newPassword"),
        )
        if not success:
            return jsonify({"success": False, "message": message})
        return jsonify({"success": True})

    def generate_stream(self, camera_id):
        frame_interval = 1.0 / self.stream_max_fps if self.stream_max_fps > 0 else 0
        last_sent_at = 0.0
        while True:
            self.start_event.wait()
            frame = self.cam_stream.get(camera_id)
            if frame is None:
                continue
            if frame_interval > 0:
                now = time.time()
                elapsed = now - last_sent_at
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_sent_at = time.time()
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n\r\n"

    def start_detection(self):
        self.start_recognition.set()
        self.detection_started = True
        return jsonify({"success": True})

    def stop_detection(self):
        self.start_recognition.clear()
        self.photo_detection_event.clear()
        self.detection_started = False
        return jsonify({"success": True})

    def start_connection(self):
        self.start_stream.set()
        self.connection_started = True
        return jsonify({"success": True})

    def reset(self):
        self.start_stream.clear()
        self.start_recognition.clear()
        self.detection_started = False
        self.connection_started = False
        return jsonify({"success": True})

    def restart_application(self):
        if self.restart_in_progress:
            return jsonify({"success": False, "message": "Application restart already in progress"}), 409
        self.restart_in_progress = True
        self.restart_requested.set()
        restart_thread = threading.Thread(target=self._delayed_server_stop, daemon=True)
        restart_thread.start()
        return jsonify({"success": True, "message": "Application restart scheduled"})

    def shutdown_application(self):
        if self.shutdown_in_progress:
            return jsonify({"success": False, "message": "Application shutdown already in progress"}), 409
        self.shutdown_in_progress = True
        self.shutdown_requested.set()
        shutdown_thread = threading.Thread(target=self._delayed_server_stop, daemon=True)
        shutdown_thread.start()
        return jsonify({"success": True, "message": "Application shutdown scheduled"})

    def snapshot(self):
        pic_path = self._get_photo_dir()
        images = []
        if pic_path and os.path.isdir(pic_path):
            for file in os.listdir(pic_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    images.append({"name": file, "url": url_for("photo", filename=file)})
        images.sort(key=lambda x: x["name"], reverse=True)
        return jsonify({"images": images})

    def serve_photo(self, filename):
        pic_path = self._get_photo_dir()
        return send_from_directory(pic_path, filename)

    def cameraBtnStatus(self):
        data = self.config.read_cameras()
        cameras = data.get("cameras", [])
        configured = any(cam.get("source") not in [None, "", " "] or cam.get("rtsp_url") not in [None, "", " "] for cam in cameras)
        data["configured"] = configured
        data["button_color"] = "green" if configured else "yellow"
        return jsonify(data)

    def botBtnStatus(self):
        data = self.config.read_bot()
        self.bot_configured = data.get("configured", False)
        return jsonify(data)

    def saveGenSettings(self):
        data = request.get_json() or {}
        self.config.save_settings(data)
        self.scheduler.apply_settings(data)
        self.stream_max_fps = self._parse_positive_int(data.get("stream_max_fps"), default=self.stream_max_fps)
        return jsonify({"success": True})

    def getGenSettings(self):
        data = self.config.read_settings()
        self.scheduler.apply_settings(data)
        self.stream_max_fps = self._parse_positive_int(data.get("stream_max_fps"), default=self.stream_max_fps)
        return jsonify(data)

    def _delayed_server_stop(self):
        time.sleep(1.0)
        self.start_stream.clear()
        self.start_recognition.clear()
        self.photo_detection_event.clear()
        self.detection_started = False
        self.connection_started = False
        self.stop()

    def stop(self):
        if self.http_server is not None:
            print("Stopping HTTP server...", flush=True)
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None

    def _read_stream_max_fps(self):
        settings = self.config.read_settings()
        return self._parse_positive_int(settings.get("stream_max_fps"), default=30)

    def _get_photo_dir(self):
        settings = self.config.read_settings()
        photo_dir = str(settings.get("photo_save_dir", "")).strip()
        if photo_dir:
            return photo_dir
        return os.path.join(self.app.root_path, "photos")

    @staticmethod
    def _parse_positive_int(value, default):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
