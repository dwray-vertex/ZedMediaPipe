#!/usr/bin/env python3
import argparse
import json
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyzed.sl as sl

DEFAULT_CALIB = "/home/user/Documents/Ultraleap_ChArUco_Python_Calibration/camera_calibration.json"

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]

FINGER_GROUPS = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", default=DEFAULT_CALIB)
    p.add_argument("--resolution", default="HD1080",
                   choices=["HD2K", "HD1080", "VGA"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--detect-width", type=int, default=640)
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2])
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1])
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)

    cams = data.get("cameras", {})
    if not cams:
        raise RuntimeError("Calibration JSON has no 'cameras' object.")

    cam = cams.get("main")
    if cam is None:
        name, cam = next(iter(cams.items()))
        print(f"[info] No 'main' camera; using '{name}'.")

    if "serial" not in cam:
        raise RuntimeError("Selected calibration camera has no serial.")

    return int(cam["serial"])


class State:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.frame = None
        self.result = None
        self.new_frame = threading.Event()
        self.stop = threading.Event()
        self.capture_fps = 0.0
        self.tracking_fps = 0.0


class CaptureThread(threading.Thread):
    def __init__(self, camera, state):
        super().__init__(daemon=True)
        self.camera = camera
        self.state = state
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.runtime = sl.RuntimeParameters()

    def run(self):
        n = 0
        t0 = time.monotonic()

        while not self.state.stop.is_set():
            if self.camera.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

            frame = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR).copy()
            depth = self.depth.get_data().copy()

            with self.state.frame_lock:
                self.state.frame = {
                    "image": frame,
                    "depth": depth,
                    "timestamp": time.monotonic(),
                }

            self.state.new_frame.set()
            n += 1

            if n >= 30:
                elapsed = time.monotonic() - t0
                self.state.capture_fps = n / elapsed if elapsed else 0.0
                n = 0
                t0 = time.monotonic()


class TrackingThread(threading.Thread):
    def __init__(self, state, width, max_hands, det_conf, track_conf, complexity):
        super().__init__(daemon=True)
        self.state = state
        self.width = width
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def run(self):
        n = 0
        t0 = time.monotonic()

        try:
            while not self.state.stop.is_set():
                self.state.new_frame.wait(0.05)
                if self.state.stop.is_set():
                    break

                with self.state.frame_lock:
                    item = self.state.frame

                if item is None:
                    continue

                frame = item["image"]
                h, w = frame.shape[:2]

                if self.width and w > self.width:
                    scale = self.width / float(w)
                    inp = cv2.resize(
                        frame,
                        (self.width, max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    inp = frame

                rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.hands.process(rgb)
                rgb.flags.writeable = True

                # Newest result only. Old results are never queued.
                with self.state.result_lock:
                    self.state.result = {
                        "results": results,
                        "image": frame,
                        "depth": item["depth"],
                        "timestamp": item["timestamp"],
                    }

                n += 1
                if n >= 30:
                    elapsed = time.monotonic() - t0
                    self.state.tracking_fps = n / elapsed if elapsed else 0.0
                    n = 0
                    t0 = time.monotonic()
        finally:
            self.hands.close()


def depth_at(depth, x, y):
    h, w = depth.shape[:2]
    x, y = int(round(x)), int(round(y))

    if not (0 <= x < w and 0 <= y < h):
        return None

    z = float(depth[y, x])
    if np.isfinite(z) and z > 0:
        return z

    vals = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            xx, yy = x + dx, y + dy
            if 0 <= xx < w and 0 <= yy < h:
                z = float(depth[yy, xx])
                if np.isfinite(z) and z > 0:
                    vals.append(z)

    return float(np.median(vals)) if vals else None


def build_finger_dict(points3d):

    if points3d is None:
        return None

    out = {}

    landmarks = points3d.landmark
    
    for finger, indices in FINGER_GROUPS.items():
        out[finger] = [
            {
                landmarks[idx].x * 100.0,
                landmarks[idx].y* 100.0, 
                landmarks[idx].z * 100.0
            }  for idx in indices]

    return out


def send_message(payload: dict):

    pass


def output_landmarks(results, depth, w, h):
    if not results or not results.multi_hand_world_landmarks:
        return

    for i, hand in enumerate(results.multi_hand_world_landmarks):
        label = "Unknown"
        if i < len(results.multi_handedness):
            label = "Right" if (results.multi_handedness[i].classification[0].label == "Left") else "Left"

        message_payload = {"hands": {label: build_finger_dict(hand)}}

        print(message_payload)
        #send_message(message_payload)


def main():
    a = args()
    serial = load_calibration(a.calib)

    resolutions = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }

    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = resolutions[a.resolution]
    init.camera_fps = a.fps
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER

    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED X Mini open failed: {status}")

    info = camera.get_camera_information()
    cfg = info.camera_configuration
    print(
        f"[info] Opened {info.camera_model}: "
        f"{cfg.resolution.width}x{cfg.resolution.height} @ {cfg.fps} FPS"
    )
    print(f"[info] Serial: {serial}")
    print("[info] Single-camera mode: no triangulation, no other cameras.")

    state = State()
    capture = CaptureThread(camera, state)
    tracking = TrackingThread(
        state,
        a.detect_width,
        a.max_hands,
        a.min_detection_confidence,
        a.min_tracking_confidence,
        a.model_complexity,
    )

    capture.start()
    tracking.start()

    last_print = 0.0

    try:
        while not state.stop.is_set():
            with state.frame_lock:
                current = state.frame

            with state.result_lock:
                result = state.result

            if current is None:
                time.sleep(0.001)
                continue

            # Display the newest captured image, not the slower MediaPipe frame.
            display = current["image"].copy()

            if result is not None:
                mp_results = result["results"]
                mp.solutions.drawing_utils.draw_landmarks(
                    display,
                    mp_results.multi_hand_landmarks[0]
                    if mp_results.multi_hand_landmarks
                    else None,
                    mp.solutions.hands.HAND_CONNECTIONS,
                ) if mp_results and mp_results.multi_hand_landmarks else None

                if mp_results and mp_results.multi_hand_landmarks:
                    # Draw all hands.
                    drawing = mp.solutions.drawing_utils
                    for hand in mp_results.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            display,
                            hand,
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )

                    now = time.monotonic()
                    if now - last_print >= 1.0:
                        h, w = display.shape[:2]
                        output_landmarks(
                            mp_results,
                            result["depth"],
                            w,
                            h,
                        )
                        last_print = now

            if not a.no_display:
                cv2.putText(
                    display,
                    f"Capture: {state.capture_fps:.1f} FPS",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"MediaPipe: {state.tracking_fps:.1f} FPS",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("ZED X Mini + MediaPipe Hands", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    state.stop.set()
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[info] Ctrl+C received.")

    finally:
        state.stop.set()
        state.new_frame.set()

        tracking.join(timeout=5)
        capture.join(timeout=5)

        camera.close()

        if not a.no_display:
            cv2.destroyAllWindows()

        print("[info] Done.")


if __name__ == "__main__":
    main()
