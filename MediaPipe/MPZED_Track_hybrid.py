"""
Usage tip: --use-roi is off by default. 
At SVGA the body-tracking overhead costs more than the smaller MediaPipe crop saves. 
Turn it on once --resolution set to (HD1080/HD1200) and want MediaPipe fed a smaller frame
"""
import argparse
import json
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyzed.sl as sl

DEFAULT_CALIB = "/home/user/Documents/Ultraleap_ChArUco_Python_Calibration/camera_calibration.json"

FINGER_GROUPS = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", default=DEFAULT_CALIB)
    p.add_argument(
        "--resolution",
        default="SVGA",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
    )
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--detect-width", type=int, default=480)
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2])
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.add_argument("--model-complexity", type=int, default=0, choices=[0, 1])

    # Off by default -- only worth it once resolution is high enough that
    # shrinking MediaPipe's input outweighs the body-detection overhead.
    p.add_argument("--use-roi", action="store_true")
    p.add_argument("--roi-padding", type=float, default=0.25)
    p.add_argument("--roi-hold", type=int, default=10)

    p.add_argument("--no-display", action="store_true")

    # OpenCV's internal thread pool (TBB) competes with MediaPipe's own
    # internal inference threads for the same CPU cores. Capping it low
    # often nets a real MediaPipe fps gain, since resize/cvtColor here are
    # cheap compared to inference. Try 1 or 2 first; 0 leaves OpenCV's
    # default (usually = core count) which is often the worse choice here.
    p.add_argument("--cv-threads", type=int, default=1)

    # --engine legacy uses mp.solutions.hands (CPU-only inference, no GPU
    # delegate exists for it on desktop -- this is the ~19.5fps ceiling).
    # --engine tasks uses the newer HandLandmarker Task API, which supports
    # a real GPU delegate on Linux and is the actual fix for hitting 24+fps
    # with 2 hands. Requires a downloaded .task model file (see --model-path).
    p.add_argument("--engine", default="legacy", choices=["legacy", "tasks"])
    p.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"])
    p.add_argument(
        "--model-path",
        default="./hand_landmarker.task",
        help=(
            "Only used with --engine tasks. Download once from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
    )

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

    return int(cam["serial"]), cam


def try_json_intrinsics(cam):
    """
    Best-effort extraction of fx/fy/cx/cy from the calibration JSON's
    camera entry. Tries a few common shapes since the exact schema of
    camera_calibration.json isn't known here:

      {"fx": .., "fy": .., "cx": .., "cy": ..}
      {"intrinsics": {"fx": .., "fy": .., "cx": .., "cy": ..}}
      {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
      {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}

    Returns None if nothing recognizable is found, so the caller can fall
    back to the ZED SDK's own (resolution-correct) intrinsics instead.
    """
    def from_flat(d):
        if all(k in d for k in ("fx", "fy", "cx", "cy")):
            return {
                "fx": float(d["fx"]), "fy": float(d["fy"]),
                "cx": float(d["cx"]), "cy": float(d["cy"]),
            }
        return None

    direct = from_flat(cam)
    if direct:
        return direct

    for key in ("intrinsics", "left_cam", "left"):
        sub = cam.get(key)
        if isinstance(sub, dict):
            found = from_flat(sub)
            if found:
                return found

    for key in ("camera_matrix", "K", "intrinsic_matrix"):
        m = cam.get(key)
        if m and len(m) == 3 and len(m[0]) == 3:
            return {
                "fx": float(m[0][0]), "fy": float(m[1][1]),
                "cx": float(m[0][2]), "cy": float(m[1][2]),
            }

    return None


def zed_intrinsics(camera):
    """
    Intrinsics as reported by the ZED SDK for the LEFT camera at the
    currently configured resolution -- always resolution-correct since
    it reflects the actual rectified image being retrieved.
    """
    calib = camera.get_camera_information().camera_configuration.calibration_parameters
    left = calib.left_cam
    return {"fx": float(left.fx), "fy": float(left.fy), "cx": float(left.cx), "cy": float(left.cy)}


def unproject(px, py, z_m, intr):
    """
    Pinhole back-projection: pixel + depth (Z, meters, along camera's
    optical axis) -> (X, Y, Z) in meters, camera optical center as origin.
    This is the actual "offset from the camera" the calibration intrinsics
    are for.
    """
    if z_m is None:
        return None, None, None
    x = (px - intr["cx"]) * z_m / intr["fx"]
    y = (py - intr["cy"]) * z_m / intr["fy"]
    return x, y, z_m


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
        self.roi_fps = 0.0

        self.roi = None
        self.roi_frame_id = -1
        self.frame_id = 0


class CaptureThread(threading.Thread):
    """
    ZED acquisition thread. Image/depth are retrieved GPU-side, then synced
    to CPU once (MediaPipe/OpenCV need CPU numpy arrays). Body-tracking
    retrieval (used only to derive a hand-search ROI) happens in this same
    thread right after grab(), so it never races the tracking thread.
    """

    def __init__(self, camera, state, use_roi, roi_padding, roi_hold):
        super().__init__(daemon=True)
        self.camera = camera
        self.state = state
        self.use_roi = use_roi
        self.roi_padding = roi_padding
        self.roi_hold = roi_hold

        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.runtime = sl.RuntimeParameters()

        self.bodies = sl.Bodies()
        self.body_runtime = sl.BodyTrackingRuntimeParameters()
        self.body_runtime.detection_confidence_threshold = 40

        self.last_roi = None
        self.last_detection_frame = -1

        self._roi_n = 0
        self._roi_t0 = time.monotonic()

    @staticmethod
    def bbox_from_body(body, width, height, padding):
        pts = np.asarray(
            [[float(p[0]), float(p[1])] for p in body.bounding_box_2d],
            dtype=np.float32,
        )
        if pts.size == 0:
            return None

        x0 = float(np.min(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        x1 = float(np.max(pts[:, 0]))
        y1 = float(np.max(pts[:, 1]))

        bw = x1 - x0
        bh = y1 - y0
        if bw <= 1 or bh <= 1:
            return None

        x0 -= bw * padding
        x1 += bw * padding
        y0 -= bh * padding
        y1 += bh * padding

        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(width, int(x1))
        y1 = min(height, int(y1))

        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def update_roi(self, frame_id, width, height):
        status = self.camera.retrieve_bodies(self.bodies, self.body_runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            return

        boxes = []
        for body in self.bodies.body_list:
            valid = True
            try:
                valid = body.tracking_state == sl.OBJECT_TRACKING_STATE.OK
            except Exception:
                pass
            if not valid:
                continue

            roi = self.bbox_from_body(body, width, height, self.roi_padding)
            if roi is not None:
                boxes.append(roi)

        if boxes:
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes)
            y1 = max(b[3] for b in boxes)
            self.last_roi = (x0, y0, x1, y1)
            self.last_detection_frame = frame_id
        elif (
            self.last_roi is not None
            and frame_id - self.last_detection_frame <= self.roi_hold
        ):
            pass
        else:
            self.last_roi = None

        with self.state.frame_lock:
            self.state.roi = self.last_roi
            self.state.roi_frame_id = self.last_detection_frame

        self._roi_n += 1
        if self._roi_n >= 30:
            elapsed = time.monotonic() - self._roi_t0
            self.state.roi_fps = self._roi_n / elapsed if elapsed > 0 else 0.0
            self._roi_n = 0
            self._roi_t0 = time.monotonic()

    def run(self):
        n = 0
        t0 = time.monotonic()

        while not self.state.stop.is_set():
            if self.camera.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            # GPU-resident retrieval + one explicit sync to CPU, instead of
            # letting the SDK do CPU retrieval + conversion on this thread.
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.GPU)
            self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH, sl.MEM.GPU)
            self.image.update_cpu_from_gpu()
            self.depth.update_cpu_from_gpu()

            frame = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR).copy()
            depth = self.depth.get_data().copy()

            frame_id = self.state.frame_id
            h, w = frame.shape[:2]

            if self.use_roi:
                self.update_roi(frame_id, w, h)

            with self.state.frame_lock:
                self.state.frame = {
                    "image": frame,
                    "depth": depth,
                    "timestamp": time.monotonic(),
                    "frame_id": frame_id,
                }
                self.state.frame_id += 1

            self.state.new_frame.set()

            n += 1
            if n >= 30:
                elapsed = time.monotonic() - t0
                self.state.capture_fps = n / elapsed if elapsed > 0 else 0.0
                n = 0
                t0 = time.monotonic()


class _TasksResultShim:
    """
    Wraps a mediapipe.tasks HandLandmarkerResult so it exposes the same
    .multi_hand_landmarks / .multi_handedness shape as the legacy
    mp.solutions.hands result. Lets build_finger_dict/draw_results/
    output_landmarks stay engine-agnostic. Tasks' NormalizedLandmark
    objects already carry .x/.y/.z, so no per-point copy is needed.
    """

    def __init__(self, task_result):
        if not task_result.hand_landmarks:
            self.multi_hand_landmarks = None
            self.multi_handedness = None
            return

        self.multi_hand_landmarks = []
        for lm_list in task_result.hand_landmarks:
            hand = type("Hand", (), {})()
            hand.landmark = lm_list
            self.multi_hand_landmarks.append(hand)

        self.multi_handedness = []
        for cat_list in task_result.handedness:
            h = type("Handedness", (), {})()
            classification = type("Classification", (), {})()
            classification.label = cat_list[0].category_name
            h.classification = [classification]
            self.multi_handedness.append(h)


class TrackingThread(threading.Thread):
    """
    Consumes only the newest frame (no queue). If a ZED body ROI is
    available, MediaPipe only sees that crop, resized to --detect-width.
    Otherwise it sees the full frame, resized the same way.

    Supports two inference engines:
      legacy -> mp.solutions.hands. CPU-only, no GPU delegate available.
      tasks  -> mediapipe.tasks HandLandmarker. Supports a real GPU
                delegate on Linux -- this is the actual lever for pushing
                past the legacy engine's CPU-bound fps ceiling.
    """

    def __init__(
        self, state, width, max_hands, det_conf, track_conf, complexity,
        engine="legacy", delegate="cpu", model_path=None,
    ):
        super().__init__(daemon=True)
        self.state = state
        self.width = width
        self.engine = engine
        self.last_frame_id = -1
        self._start = time.monotonic()

        if engine == "legacy":
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                model_complexity=complexity,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        else:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions, RunningMode,
            )

            if delegate == "gpu":
                print(
                    "[warn] --delegate gpu requires a MediaPipe build compiled "
                    "with GPU support enabled. The standard 'pip install "
                    "mediapipe' wheel on Linux does NOT include this -- it will "
                    "raise NotImplementedError / 'GPU processing is disabled in "
                    "build flags'. This requires building MediaPipe from source "
                    "with Bazel GPU flags, which is out of scope here. Falling "
                    "back to delegate=CPU."
                )
                delegate = "cpu"

            base_options = BaseOptions(
                model_asset_path=model_path,
                delegate=(
                    BaseOptions.Delegate.GPU
                    if delegate == "gpu"
                    else BaseOptions.Delegate.CPU
                ),
            )
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self.landmarker = HandLandmarker.create_from_options(options)

    def _infer(self, rgb):
        if self.engine == "legacy":
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True
            return results

        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts_ms = int((time.monotonic() - self._start) * 1000)
        task_result = self.landmarker.detect_for_video(img, ts_ms)
        return _TasksResultShim(task_result)

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
                    roi = self.state.roi

                if item is None:
                    continue

                frame = item["image"]
                frame_id = item["frame_id"]

                if frame_id == self.last_frame_id:
                    continue
                self.last_frame_id = frame_id

                h, w = frame.shape[:2]

                # FIX: x0/y0/x1/y1 must be defined on BOTH branches. This is
                # the source of the original UnboundLocalError -- roi is
                # None on every startup frame and any time it's lost.
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    x0 = max(0, min(w - 1, x0))
                    y0 = max(0, min(h - 1, y0))
                    x1 = max(x0 + 1, min(w, x1))
                    y1 = max(y0 + 1, min(h, y1))
                    inp = frame[y0:y1, x0:x1]
                else:
                    x0, y0, x1, y1 = 0, 0, w, h
                    inp = frame

                ih, iw = inp.shape[:2]  # crop size BEFORE resize -- this is
                                        # what normalized coords map back onto

                if self.width and iw > self.width:
                    scale = self.width / float(iw)
                    inp = cv2.resize(
                        inp,
                        (self.width, max(1, int(ih * scale))),
                        interpolation=cv2.INTER_AREA,
                    )

                rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                results = self._infer(rgb)

                with self.state.result_lock:
                    self.state.result = {
                        "results": results,
                        "image": frame,
                        "depth": item["depth"],
                        "timestamp": item["timestamp"],
                        "frame_id": frame_id,
                        "roi": (x0, y0, x1, y1),
                        "roi_width": iw,
                        "roi_height": ih,
                    }

                n += 1
                if n >= 30:
                    elapsed = time.monotonic() - t0
                    self.state.tracking_fps = n / elapsed if elapsed > 0 else 0.0
                    n = 0
                    t0 = time.monotonic()
        finally:
            if self.engine == "legacy":
                self.hands.close()
            else:
                self.landmarker.close()


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
                zz = float(depth[yy, xx])
                if np.isfinite(zz) and zz > 0:
                    vals.append(zz)

    return float(np.median(vals)) if vals else None


def landmark_to_original_pixel(lm, result):
    """Map a MediaPipe normalized landmark back to the native ZED pixel."""
    x0, y0, x1, y1 = result["roi"]
    roi_w = result["roi_width"]
    roi_h = result["roi_height"]
    x = x0 + float(lm.x) * roi_w
    y = y0 + float(lm.y) * roi_h
    return x, y


def build_finger_dict(hand_landmarks, result, depth, intr):
    """
    Ordered [X, Y, Z] per landmark in cm, camera optical center as origin
    -- FIX #2 (list, not a set, so order/values are guaranteed to survive)
    combined with real intrinsics-based unprojection instead of either
    frame-percentage x/y or MediaPipe's unitless landmark.z.

    X, Y: true metric offset from the camera, computed via the pinhole
          model using the calibrated fx/fy/cx/cy.
    Z:    real ZED depth along the camera's optical axis, sampled at the
          landmark's true (ROI-remapped) pixel location.
    """
    if hand_landmarks is None:
        return None

    out = {}
    landmarks = hand_landmarks.landmark

    for finger, indices in FINGER_GROUPS.items():
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            px, py = landmark_to_original_pixel(lm, result)
            z_m = depth_at(depth, px, py)
            x_m, y_m, z_m = unproject(px, py, z_m, intr)
            if x_m is None:
                pts.append([None, None, None])
            else:
                pts.append([x_m * 100.0, y_m * 100.0, z_m * 100.0])
        out[finger] = pts

    return out


def send_message(payload: dict):
    pass


def output_landmarks(results, depth, result, intr):
    """
    Builds ONE combined payload per call with both hands (when present)
    grouped under a single "hands" key, instead of a separate payload per
    hand. Note: this function is only invoked once per second (gated by
    last_print in main()), so it was never the cause of the ~19.5fps
    MediaPipe ceiling -- that's inference cost, addressed via --detect-width
    and --cv-threads (see main()).
    """
    if not results or not results.multi_hand_landmarks:
        return

    hands_payload = {}
    for i, hand in enumerate(results.multi_hand_landmarks):
        label = "Unknown"
        if i < len(results.multi_handedness):
            # Intentional inversion: the ZED looks down at the hands from
            # above rather than facing the user, which mirrors MediaPipe's
            # handedness call relative to its usual front-facing assumption.
            raw = results.multi_handedness[i].classification[0].label
            label = "Right" if raw == "Left" else "Left"

        hands_payload[label] = build_finger_dict(hand, result, depth, intr)

    message_payload = {"hands": hands_payload}
    print(message_payload)
    send_message(message_payload)


def draw_results(display, result):
    results = result["results"]
    if not results or not results.multi_hand_landmarks:
        return

    x0, y0, x1, y1 = result["roi"]
    roi_w = result["roi_width"]
    roi_h = result["roi_height"]

    for hand in results.multi_hand_landmarks:
        points = []
        for lm in hand.landmark:
            x = int(x0 + lm.x * roi_w)
            y = int(y0 + lm.y * roi_h)
            points.append((x, y))

        for a, b in mp.solutions.hands.HAND_CONNECTIONS:
            if 0 <= a < len(points) and 0 <= b < len(points):
                cv2.line(display, points[a], points[b], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)

    cv2.rectangle(display, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)


def main():
    a = args()
    serial, cam_info = load_calibration(a.calib)

    cv2.setNumThreads(a.cv_threads)
    print(f"[info] cv2.setNumThreads({a.cv_threads}) -- lower reduces OpenCV/MediaPipe thread contention")

    resolutions = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
    }

    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = resolutions[a.resolution]
    init.camera_fps = a.fps
    init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init.coordinate_units = sl.UNIT.METER

    camera = sl.Camera()
    status = camera.open(init)

    camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 40)
    camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 10)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED X Mini open failed: {status}")

    info = camera.get_camera_information()
    cfg = info.camera_configuration
    print(
        f"[info] Opened {info.camera_model}: "
        f"{cfg.resolution.width}x{cfg.resolution.height} @ {cfg.fps} FPS"
    )
    print(f"[info] Serial: {serial}")
    print("[info] ZED retrieval memory: GPU")
    print(f"[info] Body-ROI cropping: {'ON' if a.use_roi else 'OFF'}")
    print(f"[info] MediaPipe engine: {a.engine}" + (f" (delegate={a.delegate})" if a.engine == "tasks" else " (CPU only)"))

    intr = try_json_intrinsics(cam_info)
    if intr is not None:
        print(f"[info] Intrinsics source: calibration JSON  fx={intr['fx']:.2f} fy={intr['fy']:.2f} cx={intr['cx']:.2f} cy={intr['cy']:.2f}")
    else:
        intr = zed_intrinsics(camera)
        print(
            "[warn] Could not find fx/fy/cx/cy in the calibration JSON "
            "(checked flat keys, 'intrinsics'/'left_cam', and 'camera_matrix'/'K'). "
            "Falling back to the ZED SDK's own reported intrinsics for the "
            "current resolution -- tell me your JSON's actual key names if "
            "you want it to use the ChArUco calibration's values instead."
        )
        print(f"[info] Intrinsics source: ZED SDK  fx={intr['fx']:.2f} fy={intr['fy']:.2f} cx={intr['cx']:.2f} cy={intr['cy']:.2f}")

    if a.use_roi:
        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = False
        body_params.enable_body_fitting = False
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_params.body_format = sl.BODY_FORMAT.BODY_18

        status = camera.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED body tracking enable failed: {status}")

    state = State()
    capture = CaptureThread(camera, state, a.use_roi, a.roi_padding, a.roi_hold)
    tracking = TrackingThread(
        state,
        a.detect_width,
        a.max_hands,
        a.min_detection_confidence,
        a.min_tracking_confidence,
        a.model_complexity,
        engine=a.engine,
        delegate=a.delegate,
        model_path=a.model_path,
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

            if not a.no_display:
                display = current["image"].copy()

                if result is not None:
                    mp_results = result["results"]
                    if mp_results and mp_results.multi_hand_landmarks:
                        draw_results(display, result)

                        now = time.monotonic()
                        if now - last_print >= 1.0:
                            output_landmarks(mp_results, result["depth"], result, intr)
                            last_print = now

                cv2.putText(
                    display, f"Capture: {state.capture_fps:.1f} FPS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.putText(
                    display, f"MediaPipe: {state.tracking_fps:.1f} FPS", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                if a.use_roi:
                    cv2.putText(
                        display, f"Body ROI: {state.roi_fps:.1f} FPS", (15, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                    )

                cv2.imshow("ZED X Mini + MediaPipe Hands (hybrid)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    state.stop.set()
                    break
            else:
                if result is not None:
                    mp_results = result["results"]
                    now = time.monotonic()
                    if now - last_print >= 1.0:
                        output_landmarks(mp_results, result["depth"], result, intr)
                        print(f"Capture: {state.capture_fps:.1f} FPS")
                        print(f"MediaPipe: {state.tracking_fps:.1f} FPS")
                        last_print = now
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[info] Ctrl+C received.")

    finally:
        state.stop.set()
        state.new_frame.set()

        tracking.join(timeout=5)
        capture.join(timeout=5)

        if a.use_roi:
            try:
                camera.disable_body_tracking()
            except Exception:
                pass

        camera.close()

        if not a.no_display:
            cv2.destroyAllWindows()

        print("[info] Done.")


if __name__ == "__main__":
    main()
