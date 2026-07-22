# Surveillance Project

This directory contains a local IP-camera surveillance application. 
It captures camera streams, runs YOLO object detection, displays live video in a Flask web dashboard, 
saves snapshots, and can send Telegram alerts when motion and target objects are detected.

The scope of this README is only the `survilance` directory.

## Main Features

- Web dashboard for camera monitoring and configuration.
- RTSP/IP camera capture with OpenCV.
- YOLO-based object detection for classes such as `person` and `car`.
- Motion comparison between frames to reduce unnecessary alerts.
- Snapshot saving to a configured photos directory.
- Telegram photo alerts.
- Basic login and password change support.
- Camera, bot, and general settings stored as JSON files.
- Optional automatic scheduled start and old-photo cleanup settings.

## Main Entry Point

Run the application from this directory with:

```powershell
python start_recognition.py
```

The script starts `CameraPipeline`, which coordinates:

- `Server.py`: Flask dashboard and API.
- `camera.py`: camera stream capture.
- `recognition.py`: YOLO detection worker.
- `photo_rec.py`: snapshot writer.
- `messanger.py`: Telegram notification sender.

The Flask server currently starts on port `8080`.

## Project Structure

```text
survilance/
  start_recognition.py       Main launcher
  Pipeline.py                Starts and coordinates server, cameras, detection, snapshots, Telegram
  Server.py                  Flask web server, dashboard routes, API routes, auth, settings
  camera.py                  Active camera capture class
  recognition.py             YOLO detection worker
  photo_rec.py               Saves frames as image files
  messanger.py               Sends Telegram photo alerts
  config_store.py            JSON-backed camera, bot, and settings storage
  auth_store.py              SQLite-backed dashboard user storage
  scheduler.py               Scheduled start and old snapshot cleanup jobs
  templates/
    surveillance_dashboard.html
  system_settings/
    cameras.json             Camera list and sources
    bot.json                 Telegram bot settings
    settings.json            Schedule, photo path, cleanup settings
  <path_to_photo_dir>/       Saved snapshots
```

## Configuration Files

The active pipeline uses files in `system_settings/`.

### `system_settings/cameras.json`

Stores camera definitions:

```json
{
  "cameras": [
    {
      "id": 0,
      "name": "CAM 1",
      "source": "rtsp://user:password@camera-ip:port/stream"
    }
  ]
}
```

The `source` value can be an RTSP URL or a local camera index such as `0`.

### `system_settings/bot.json`

Stores Telegram bot configuration:

```json
{
  "token": "",
  "api_url": "",
  "username": "",
  "chat_id": ""
}
```

Do not commit real bot tokens or chat IDs.

### `system_settings/settings.json`

Stores general runtime settings:

```json
{
  "schedule_time": "--:--",
  "photo_save_dir": "C:/path/to/photos",
  "cleanup_days": null
}
```

## Dashboard/API Overview

The Flask server provides:

- `/`: surveillance dashboard.
- `/login`: login endpoint.
- `/change-credentials`: update dashboard credentials.
- `/api/cameras` and `/api/cameras/save`: read/save camera settings.
- `/api/bot` and `/api/bot/save`: read/save Telegram bot settings.
- `/stream/<camera_id>`: MJPEG camera stream.
- `/api/detection/connect`: start camera stream capture.
- `/api/detection/start`: start detection.
- `/api/detection/stop`: stop detection.
- `/api/detection/reset`: clear stream/detection state.
- `/api/snapshots`: list saved images.
- `/photo/<filename>`: serve saved image files.
- `/api/saveGenSettings` and `/api/getSettings`: save/read general settings.
- `/api/getInfo`: get current dashboard status information.

## Dependencies

The code uses:

- Python
- OpenCV (`cv2`)
- Flask
- Werkzeug
- Ultralytics YOLO
- PyTorch
- cvzone
- requests
- NumPy
- local `yolov5` package/module access

The current repository has a virtual environment one level above this directory in `.venv`. If using that environment from `survilance`, commands look like:

```powershell
..\.venv\Scripts\python.exe start_recognition.py
```

## Runtime Notes

- The default dashboard credentials are created in `users.db` as `admin` / `admin` if no user exists.
- `Server.py` currently binds to a fixed host IP and port `8080`. 
- If the server does not start on another machine, update the `host` value in `Server.run()`.
- The model path used by detection is `Ip_cam_survilance/yolo11n.pt`.
- RTSP transport is set to TCP in `start_recognition.py` through `OPENCV_FFMPEG_CAPTURE_OPTIONS`.
- Generated files such as `.db` files, snapshots, and local credentials should normally stay out of version control.

## Security Notes

This project stores camera URLs, Telegram tokens, chat IDs, and dashboard credentials locally. Treat these files as private:

- `Bot_credentials.py`
- `system_settings/bot.json`
- `system_settings/cameras.json`
- `users.db`


Before sharing or committing the project, remove real tokens, passwords, chat IDs, and RTSP credentials.

full topic for reading here bellow:
https://www.hackster.io/Dima_Panasenko/surveillance-system-with-computer-256d6a
