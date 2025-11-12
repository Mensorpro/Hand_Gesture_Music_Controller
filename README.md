# Hand Gesture Music Controller

Control media with your hand using OpenCV + MediaPipe. This version uses OS media keys and app hotkeys only .

## Requirements
- Windows 10/11
- Python 3.11 (recommended for MediaPipe wheels)
- Webcam

## Setup

Option A — using `uv` (fast, recommended):
```
uv python install 3.11
uv venv --python 3.11
uv pip install --python .venv -r requirements.txt
```

Option B — using `pip` directly:
```
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
```
. .\.venv\Scripts\Activate.ps1
python hand_music_control.py
```

Press ESC to exit.

## Gestures (default)
- Open palm: Play/Pause
- Index only (Left/Right hand): Previous/Next
- Four fingers up (thumb tucked): Left hand Volume Down, Right hand Volume Up
- Pinky only: Repeat toggle
- Peace (index+middle): Shuffle toggle
- Shaka (thumb+pinky): Mute/Unmute
- Open palm swipe left/right: Seek backward/forward (Ctrl+Arrow)

## Notes
- Shuffle/Repeat use app hotkeys (Ctrl+S / Ctrl+R). Keep Spotify Desktop focused for those to work.
- Gestures use a short hold time and cooldown to reduce misfires; tweak at the top of `hand_music_control.py`:
  - `GESTURE_HOLD_SECONDS`, `GESTURE_COOLDOWN_SECONDS`
  - Per-gesture overrides in `GESTURE_HOLD_MAP`, `GESTURE_COOLDOWN_MAP`

## Troubleshooting
- MediaPipe won’t install: ensure you’re on Python 3.11 (not 3.13). Recreate the venv with 3.11 and reinstall requirements.
- Webcam not opening: close other apps using the camera and retry; check privacy permissions in Windows Settings → Privacy & security → Camera.
- Hotkeys not working (shuffle/repeat/seek): bring the media app (e.g., Spotify Desktop) to the foreground so it receives Ctrl+S / Ctrl+R / Ctrl+Arrow.
- Gestures too sensitive: increase `GESTURE_HOLD_SECONDS` and/or `GESTURE_COOLDOWN_SECONDS`.
