import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import pystray
from PIL import Image, ImageDraw

# Disable the mouse fail-safe so gestures near corners do not raise exceptions.
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

last_trigger_time = 0.0
last_gesture = None
pending_gesture = None
pending_start_time = 0.0


WINDOW_TITLE = "Hand Gesture Music Controller"
ASSETS_DIR = Path(__file__).with_name("assets")


@dataclass
class RuntimeSettings:
    base_hold_seconds: float = 0.25
    base_cooldown_seconds: float = 1.2
    hold_multiplier: dict = field(
        default_factory=lambda: {
            "volume_up": 0.88,
            "volume_down": 0.88,
        }
    )
    cooldown_multiplier: dict = field(
        default_factory=lambda: {
            "volume_up": 0.375,
            "volume_down": 0.375,
            "mute_toggle": 0.5,
            "repeat_toggle": 0.6666666667,
            "shuffle_toggle": 0.6666666667,
        }
    )
    volume_press_count: int = 2
    volume_press_interval: float = 0.08
    volume_preset_name: str = "Balanced (2 steps)"
    sensitivity_preset_name: str = "Balanced"

    def get_hold_seconds(self, gesture_key: str) -> float:
        multiplier = self.hold_multiplier.get(gesture_key, 1.0)
        return self.base_hold_seconds * multiplier

    def get_cooldown_seconds(self, gesture_key: str) -> float:
        multiplier = self.cooldown_multiplier.get(gesture_key, 1.0)
        return self.base_cooldown_seconds * multiplier


@dataclass(frozen=True)
class PowerProfile:
    name: str
    frame_width: Optional[int] = None
    max_fps: Optional[float] = None
    model_complexity: int = 1

    def frame_interval(self) -> float:
        if self.max_fps and self.max_fps > 0:
            return 1.0 / self.max_fps
        return 0.0


settings = RuntimeSettings()
settings_lock = threading.Lock()

VOLUME_PRESETS = [
    ("Gentle (1 step)", 1, 0.12),
    ("Balanced (2 steps)", 2, 0.08),
    ("Bold (4 steps)", 4, 0.06),
]

SENSITIVITY_PRESETS = [
    ("Relaxed", 0.32, 1.6),
    ("Balanced", 0.25, 1.2),
    ("Snappy", 0.18, 0.8),
]


def set_volume_preset(name: str, presses: int, interval: float) -> None:
    with settings_lock:
        settings.volume_preset_name = name
        settings.volume_press_count = presses
        settings.volume_press_interval = interval


def set_sensitivity_preset(name: str, base_hold: float, base_cooldown: float) -> None:
    with settings_lock:
        settings.sensitivity_preset_name = name
        settings.base_hold_seconds = base_hold
        settings.base_cooldown_seconds = base_cooldown

    reset_pending_state()


def show_error_dialog(message: str) -> None:
    """Display a native Windows message box with the provided message."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, 0x10)
    except Exception:
        pass


def reset_pending_state() -> None:
    """Clear the currently pending gesture hold."""
    global pending_gesture, pending_start_time
    pending_gesture = None
    pending_start_time = 0.0


def is_spotify_active() -> bool:
    """Return True if the active window title suggests Spotify is focused."""
    try:
        window = gw.getActiveWindow()
        if not window:
            return False
        title = window.title or ""
        return "spotify" in title.lower()
    except Exception:
        return False


def fingers_up(hand_landmarks, frame_shape, handedness_label):
    """Return finger state list and landmark positions for the detected hand."""
    h, w, _ = frame_shape
    lm_positions = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    finger_states = []

    # Thumb: compare x positions, correcting for left/right hand orientation.
    thumb_tip_x, thumb_ip_x = lm_positions[4][0], lm_positions[3][0]
    if handedness_label == "Right":
        thumb_is_extended = thumb_tip_x > thumb_ip_x
    else:
        thumb_is_extended = thumb_tip_x < thumb_ip_x
    finger_states.append(1 if thumb_is_extended else 0)

    # Other four fingers: tip y above pip y (smaller pixel value) means the finger is raised.
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        tip_y = lm_positions[tip_id][1]
        pip_y = lm_positions[tip_id - 2][1]
        finger_states.append(1 if tip_y < pip_y else 0)

    return finger_states, lm_positions


def trigger_gesture(gesture_name: str) -> None:
    """Send the mapped keyboard or hotkey command via pyautogui."""
    with settings_lock:
        volume_press_count = settings.volume_press_count
        volume_press_interval = settings.volume_press_interval
    actions = {
        "play_pause": lambda: pyautogui.press("playpause"),
        "volume_up": lambda: pyautogui.press(
            "volumeup",
            presses=volume_press_count,
            interval=volume_press_interval,
        ),
        "volume_down": lambda: pyautogui.press(
            "volumedown",
            presses=volume_press_count,
            interval=volume_press_interval,
        ),
        "mute_toggle": lambda: pyautogui.press("volumemute"),
        "next_track": lambda: pyautogui.press("nexttrack"),
        "prev_track": lambda: pyautogui.press("prevtrack"),
        "shuffle_toggle": lambda: pyautogui.hotkey("ctrl", "s"),
        "repeat_toggle": lambda: pyautogui.hotkey("ctrl", "r"),
    }

    action = actions.get(gesture_name)
    if action:
        if gesture_name in {"shuffle_toggle", "repeat_toggle"} and not is_spotify_active():
            return
        action()


def classify_static_gesture(finger_state, lm_positions, handedness_label):
    """Identify static gestures based on finger states and orientation."""
    if all(finger_state):
        return "play_pause", "Play / Pause"

    if finger_state[0] == 1 and finger_state[1] == 0 and finger_state[2] == 0 and finger_state[3] == 0 and finger_state[4] == 1:
        return "mute_toggle", "Mute / Unmute"

    if finger_state[0] == 0 and all(finger_state[1:]):
        if handedness_label == "Left":
            return "volume_down", "Volume Down"
        return "volume_up", "Volume Up"

    if finger_state[4] == 1 and sum(finger_state[:4]) == 0:
        return "repeat_toggle", "Repeat Toggle"

    if (
        finger_state[0] == 0
        and finger_state[1] == 1
        and finger_state[2] == 0
        and finger_state[3] == 0
        and finger_state[4] == 0
    ):
        if handedness_label == "Left":
            return "prev_track", "Previous Song"
        return "next_track", "Next Song"

    if (
        finger_state[0] == 0
        and finger_state[1] == 1
        and finger_state[2] == 1
        and finger_state[3] == 0
        and finger_state[4] == 0
    ):
        return "shuffle_toggle", "Shuffle Toggle"

    return None, ""


def handle_gesture(gesture_key, friendly_name, now):
    """Trigger the gesture if cooldown permits and return label/trigger status."""
    global last_trigger_time, last_gesture

    if not gesture_key:
        return "", False

    triggered = False

    with settings_lock:
        cooldown = settings.get_cooldown_seconds(gesture_key)

    if gesture_key != last_gesture or now - last_trigger_time >= cooldown:
        trigger_gesture(gesture_key)
        last_gesture = gesture_key
        last_trigger_time = now
        triggered = True

    return friendly_name, triggered


class GestureController:
    """Run the hand-tracking loop on a background thread."""

    def __init__(self, show_camera: bool, *, power_profile: PowerProfile) -> None:
        self._show_preview = threading.Event()
        if show_camera:
            self._show_preview.set()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="GestureController", daemon=True)
        self._window_open = False
        self._config_lock = threading.Lock()
        self._target_width = power_profile.frame_width if power_profile.frame_width and power_profile.frame_width > 0 else None
        self._frame_interval = power_profile.frame_interval()
        model_complexity = max(0, min(2, power_profile.model_complexity))
        self._desired_model_complexity = model_complexity
        self._current_model_complexity = model_complexity
        self._active_power_profile = power_profile

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Ensure waitKey exits promptly if window is open.
        self._show_preview.set()
        self._thread.join(timeout=3.0)

    def join(self) -> None:
        self._thread.join()

    def show_preview(self) -> None:
        self._show_preview.set()

    def hide_preview(self) -> None:
        self._show_preview.clear()

    def toggle_preview(self) -> None:
        if self.is_preview_visible():
            self.hide_preview()
        else:
            self.show_preview()

    def is_preview_visible(self) -> bool:
        return self._show_preview.is_set()

    def set_power_profile(self, profile: PowerProfile) -> None:
        with self._config_lock:
            self._target_width = profile.frame_width if profile.frame_width and profile.frame_width > 0 else None
            self._frame_interval = profile.frame_interval()
            self._desired_model_complexity = max(0, min(2, profile.model_complexity))
            self._active_power_profile = profile

    def get_power_profile_name(self) -> str:
        with self._config_lock:
            return self._active_power_profile.name

    def _run(self) -> None:
        global pending_gesture, pending_start_time

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=self._current_model_complexity,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            show_error_dialog("Unable to open the webcam. Check camera permissions and availability.")
            hands.close()
            return

        try:
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                with self._config_lock:
                    target_width = self._target_width
                    frame_interval = self._frame_interval

                success, frame = cap.read()
                if not success:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                if target_width and frame.shape[1] > target_width:
                    scale = target_width / frame.shape[1]
                    new_height = max(1, int(frame.shape[0] * scale))
                    frame = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                active_label = ""

                if results.multi_hand_landmarks:
                    handedness_list = results.multi_handedness or []

                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        handedness_label = "Right"
                        if handedness_list and idx < len(handedness_list):
                            handedness_label = handedness_list[idx].classification[0].label

                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        finger_state, lm_positions = fingers_up(hand_landmarks, frame.shape, handedness_label)
                        now = time.time()

                        gesture_key, friendly_name = classify_static_gesture(
                            finger_state,
                            lm_positions,
                            handedness_label,
                        )

                        if gesture_key:
                            with settings_lock:
                                required_hold = settings.get_hold_seconds(gesture_key)
                            if pending_gesture != gesture_key:
                                pending_gesture = gesture_key
                                pending_start_time = now

                            hold_elapsed = now - pending_start_time
                            if hold_elapsed >= required_hold:
                                label, triggered = handle_gesture(gesture_key, friendly_name, now)
                                active_label = label if label else friendly_name
                                if triggered:
                                    pending_start_time = now
                            else:
                                active_label = f"Hold: {friendly_name}"
                        else:
                            reset_pending_state()
                            active_label = "Waiting for gesture..."

                        break
                else:
                    reset_pending_state()

                if not active_label:
                    active_label = "Waiting for gesture..."

                if self.is_preview_visible():
                    if not self._window_open:
                        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
                        self._window_open = True

                    cv2.putText(
                        frame,
                        active_label,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                    )

                    cv2.imshow(WINDOW_TITLE, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        self.hide_preview()
                else:
                    if self._window_open:
                        cv2.destroyWindow(WINDOW_TITLE)
                        self._window_open = False
                    time.sleep(0.01)

                if frame_interval > 0:
                    elapsed = time.perf_counter() - loop_start
                    remaining = frame_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                with self._config_lock:
                    desired_complexity = self._desired_model_complexity

                if desired_complexity != self._current_model_complexity:
                    hands.close()
                    hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        model_complexity=desired_complexity,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.6,
                    )
                    self._current_model_complexity = desired_complexity
        finally:
            cap.release()
            if self._window_open:
                cv2.destroyWindow(WINDOW_TITLE)
            hands.close()
            reset_pending_state()


def _build_tray_icon() -> Image.Image:
    """Load the tray icon from assets, falling back to a generated glyph."""
    for candidate in ("tray.png", "tray.ico", "app.ico"):
        asset_path = ASSETS_DIR / candidate
        if not asset_path.exists():
            continue
        try:
            icon_image = Image.open(asset_path)
            if icon_image.mode != "RGBA":
                icon_image = icon_image.convert("RGBA")
            return icon_image
        except Exception:
            continue

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, size - 6, size - 6), fill=(0, 180, 255, 255))
    draw.rectangle((size // 2 - 6, 16, size // 2 + 6, size - 16), fill=(255, 255, 255, 255))
    draw.rectangle((size // 2 - 18, size - 18, size // 2 + 18, size - 10), fill=(255, 255, 255, 200))
    return image


class TrayApplication:
    """System tray wrapper that controls the gesture controller lifecycle."""

    def __init__(
        self,
        controller: GestureController,
        *,
        default_profile: PowerProfile,
        low_power_profile: PowerProfile,
    ) -> None:
        self._controller = controller
        self._default_profile = default_profile
        self._low_power_profile = low_power_profile
        volume_menu = pystray.Menu(
            *[
                pystray.MenuItem(
                    label,
                    self._make_volume_handler(label, presses, interval),
                    checked=self._volume_checked(label),
                )
                for label, presses, interval in VOLUME_PRESETS
            ]
        )

        sensitivity_menu = pystray.Menu(
            *[
                pystray.MenuItem(
                    label,
                    self._make_sensitivity_handler(label, hold, cooldown),
                    checked=self._sensitivity_checked(label),
                )
                for label, hold, cooldown in SENSITIVITY_PRESETS
            ]
        )

        self._icon = pystray.Icon(
            "HandGestureMusicController",
            icon=_build_tray_icon(),
            title=WINDOW_TITLE,
            menu=pystray.Menu(
                pystray.MenuItem(
                    "Camera preview",
                    self._toggle_preview,
                    checked=lambda item: self._controller.is_preview_visible(),
                ),
                pystray.MenuItem("Volume step", volume_menu),
                pystray.MenuItem("Gesture sensitivity", sensitivity_menu),
                pystray.MenuItem(
                    "Battery saver mode",
                    self._toggle_low_power,
                    checked=self._low_power_checked,
                ),
                pystray.MenuItem("Exit", self._exit),
            ),
        )

    def run(self) -> None:
        self._icon.run()

    def stop(self) -> None:
        self._icon.stop()

    def _volume_checked(self, preset_name: str):
        def checker(item):
            with settings_lock:
                return settings.volume_preset_name == preset_name

        return checker

    def _sensitivity_checked(self, preset_name: str):
        def checker(item):
            with settings_lock:
                return settings.sensitivity_preset_name == preset_name

        return checker

    def _make_volume_handler(self, label: str, presses: int, interval: float):
        def handler(icon, item):
            set_volume_preset(label, presses, interval)

        return handler

    def _make_sensitivity_handler(self, label: str, hold: float, cooldown: float):
        def handler(icon, item):
            set_sensitivity_preset(label, hold, cooldown)

        return handler

    def _toggle_preview(self, icon, item) -> None:
        self._controller.toggle_preview()

    def _low_power_checked(self, item) -> bool:
        return self._controller.get_power_profile_name() == self._low_power_profile.name

    def _toggle_low_power(self, icon, item) -> None:
        if self._low_power_checked(item):
            self._controller.set_power_profile(self._default_profile)
        else:
            self._controller.set_power_profile(self._low_power_profile)

    def _exit(self, icon, item) -> None:
        icon.visible = False
        icon.stop()


def main(
    initial_preview: bool,
    enable_tray: bool,
    *,
    default_profile: PowerProfile,
    start_profile: PowerProfile,
    low_power_profile: PowerProfile,
) -> None:
    controller = GestureController(
        show_camera=initial_preview,
        power_profile=start_profile,
    )
    controller.start()

    try:
        if enable_tray:
            tray = TrayApplication(
                controller,
                default_profile=default_profile,
                low_power_profile=low_power_profile,
            )
            tray.run()
        else:
            try:
                controller.join()
            except KeyboardInterrupt:
                pass
    finally:
        controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control media playback with hand gestures")
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="Start with the camera preview window visible",
    )
    parser.add_argument(
        "--hide-camera",
        action="store_true",
        help="Alias for running without the preview window",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run without creating a system-tray icon (Ctrl+C to exit)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Downscale frames to this width before processing (default: camera native)",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Limit gesture processing to this many frames per second (default: unlimited)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="MediaPipe Hands model complexity (0=lite, 1=default, 2=high)",
    )
    parser.add_argument(
        "--low-power",
        action="store_true",
        help="Use energy-saving defaults (640px frames, 15 FPS cap, lite model)",
    )
    args = parser.parse_args()

    show_preview = args.show_camera and not args.hide_camera
    run_tray = not args.no_tray
    frame_width = args.frame_width if args.frame_width and args.frame_width > 0 else None
    max_fps = args.max_fps if args.max_fps and args.max_fps > 0 else None
    model_complexity = args.model_complexity if args.model_complexity is not None else 1

    overrides_requested = any(
        [
            frame_width is not None,
            max_fps is not None,
            args.model_complexity is not None,
        ]
    )

    default_profile = PowerProfile(
        name="Balanced" if not overrides_requested else "Custom",
        frame_width=frame_width,
        max_fps=max_fps,
        model_complexity=model_complexity,
    )

    low_power_profile = PowerProfile(
        name="Battery saver",
        frame_width=640,
        max_fps=15.0,
        model_complexity=0,
    )

    start_profile = low_power_profile if args.low_power else default_profile

    main(
        initial_preview=show_preview,
        enable_tray=run_tray,
        default_profile=default_profile,
        start_profile=start_profile,
        low_power_profile=low_power_profile,
    )
