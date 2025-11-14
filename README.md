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
python hand_music_control.py [--show-camera] [--no-tray] [--low-power]
```

By default the app runs headless with a system-tray icon (look for the blue hand icon in the Windows notification area). Use the tray menu to toggle **Camera preview** or **Exit** the app. Pass `--show-camera` to start with the preview window visible, or `--no-tray` for a console-only session (press Ctrl+C to stop). The legacy `--hide-camera` flag is still accepted and behaves the same as the default.

### Reduce resource usage

- `--low-power` enables a sensible trio of settings (lite MediaPipe model, 640px frames, 15 FPS cap) while keeping gesture accuracy acceptable.
- `--frame-width <pixels>` downscales each frame before MediaPipe runs (e.g. `--frame-width 800`).
- `--max-fps <number>` throttles gesture evaluation to a fixed rate (e.g. `--max-fps 20`).
- `--model-complexity {0,1,2}` lets you explicitly pick the MediaPipe Hands model tier (0 is fastest, 2 is most accurate).
- Use the tray’s **Battery saver mode** toggle to jump between your default profile (balanced or custom CLI values) and the low-power preset without restarting the app.

Combine these flags to match your hardware: for example, `python hand_music_control.py --low-power --frame-width 720` keeps GPU/CPU usage low on ultrabooks, while `--model-complexity 2 --max-fps 0` restores the maximum fidelity configuration. The tray toggle lets you switch on demand once the app is running.

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
- Gestures use short hold/cooldown windows to reduce misfires; adjust them from the tray via **Gesture sensitivity** (Relaxed/Balanced/Snappy).
- Use **Volume step** in the tray to control how many OS volume notches fire per gesture.
- A tray icon is always created unless `--no-tray` is passed. Use it to toggle the preview window, tweak sensitivity/volume, or exit without killing the process in Task Manager.

## Troubleshooting

- MediaPipe won’t install: ensure you’re on Python 3.11 (not 3.13). Recreate the venv with 3.11 and reinstall requirements.
- Webcam not opening: close other apps using the camera and retry; check privacy permissions in Windows Settings → Privacy & security → Camera.
- Hotkeys not working (shuffle/repeat/seek): bring the media app (e.g., Spotify Desktop) to the foreground so it receives Ctrl+S / Ctrl+R / Ctrl+Arrow.
- Gestures too sensitive: pick **Relaxed** from the tray’s **Gesture sensitivity** menu. Too sluggish? choose **Snappy**.

## Build the Windows executable

Package a standalone `.exe` when you’re ready to distribute:

1. Ensure the virtual environment is active and install PyInstaller:

```
uv pip install --python .venv pyinstaller
```

or

```
python -m pip install pyinstaller
```

2. Run PyInstaller with the supplied spec:

```
pyinstaller hand_music_control.spec --noconfirm
```

3. The frozen build lives in `dist/HandGestureMusicController/HandGestureMusicController.exe`. Keep `dist/` out of git (already ignored) and upload the executable to a GitHub Release instead of committing it to the repository.

`hand_music_control.spec` collects the MediaPipe/OpenCV assets that PyInstaller misses by default, so use that spec rather than the raw script path when freezing.

### Ship a GitHub release

1. Tag the repo: `git tag vX.Y.Z && git push origin vX.Y.Z`.
2. Build with PyInstaller (step 1–3 above).
3. Create a release at https://github.com/Mensorpro/Hand_Gesture_Music_Controller/releases/new, attach the `.exe` (and a `.zip` if you prefer), and include summarized changes.
4. Update the `README.md` changelog or release notes section as needed.
