import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

# Disable the mouse fail-safe so gestures near corners do not raise exceptions.
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

# Timing thresholds to smooth out gesture detection.
GESTURE_COOLDOWN_SECONDS = 1.2
GESTURE_COOLDOWN_MAP = {
    "volume_up": 0.45,
    "volume_down": 0.45,
    "mute_toggle": 0.6,
    "repeat_toggle": 0.8,
    "shuffle_toggle": 0.8,
}
GESTURE_HOLD_SECONDS = 0.25
GESTURE_HOLD_MAP = {
    "volume_up": 0.22,
    "volume_down": 0.22,
}

last_trigger_time = 0.0
last_gesture = None
pending_gesture = None
pending_start_time = 0.0


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
    actions = {
        "play_pause": lambda: pyautogui.press("playpause"),
        "volume_up": lambda: pyautogui.press("volumeup", presses=2, interval=0.08),
        "volume_down": lambda: pyautogui.press("volumedown", presses=2, interval=0.08),
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

    cooldown = GESTURE_COOLDOWN_MAP.get(gesture_key, GESTURE_COOLDOWN_SECONDS)

    if gesture_key != last_gesture or now - last_trigger_time >= cooldown:
        trigger_gesture(gesture_key)
        last_gesture = gesture_key
        last_trigger_time = now
        triggered = True

    return friendly_name, triggered


def main():
    global pending_gesture, pending_start_time

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
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
                        required_hold = GESTURE_HOLD_MAP.get(gesture_key, GESTURE_HOLD_SECONDS)
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

            cv2.putText(
                frame,
                active_label,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

            cv2.imshow("Hand Gesture Music Controller", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
