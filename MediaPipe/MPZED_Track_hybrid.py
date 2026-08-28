"""
ZED X Mini + MediaPipe Hands

3D reconstruction:
    - ZED depth provides the absolute wrist position.
    - MediaPipe provides the relative 3D hand shape.
    - Individual finger landmarks are NOT independently assigned ZED depth values.
    - Output is converted to Unreal-style coordinates:

          Unreal X = ZED Z   (forward)
          Unreal Y = ZED X   (right)
          Unreal Z = -ZED Y  (up)

--use-roi is off by default.

    At SVGA the body-tracking overhead can cost more than the smaller
    MediaPipe crop saves. Turn it on at HD1080/HD1200 when appropriate.
"""

import argparse
import json
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import pyzed.sl as sl


DEFAULT_CALIB = ("/home/user/Documents/Ultraleap_ChArUco_Python_Calibration/camera_calibration.json")

# remove later if not tracking point
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

    p.add_argument("--resolution",default="SVGA", choices=["HD2K","HD1200","HD1080","HD720","SVGA","VGA",])

    p.add_argument("--fps",type=int,default=60)
    p.add_argument("--detect-width",type=int,default=480)
    p.add_argument("--max-hands",type=int,default=2, choices=[1, 2])
    p.add_argument("--min-detection-confidence", type=float,default=0.5)
    p.add_argument("--min-tracking-confidence",type=float, default=0.5)
    p.add_argument("--model-complexity",type=int,default=0, choices=[0, 1])
    p.add_argument("--use-roi",action="store_true")
    p.add_argument("--roi-padding",type=float,default=0.25)
    p.add_argument("--roi-hold",type=int,default=10)
    p.add_argument("--no-display",action="store_true")
    p.add_argument("--cv-threads",type=int,default=2)
    p.add_argument( "--engine",default="legacy",choices=["legacy", "tasks"])
    p.add_argument("--delegate",default="cpu",choices=["cpu", "gpu"])
    p.add_argument("--model-path",default="./hand_landmarker.task",help=("Only used with --engine tasks."))

    # MediaPipe relative-hand scale.
    # Default = 1.0. Deliberately single scale for complete hand rather than
    # independent scale/depth for every finger.
    p.add_argument("--hand-scale",type=float, default=1.0, help=("Scale applied to MediaPipe wrist-relative hand geometry."))

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
    Best-effort extraction of fx/fy/cx/cy.
    """

    def from_flat(d):
        if all(
            k in d
            for k in ("fx", "fy", "cx", "cy")
        ):
            return {
                "fx": float(d["fx"]),
                "fy": float(d["fy"]),
                "cx": float(d["cx"]),
                "cy": float(d["cy"]),
            }

        return None

    direct = from_flat(cam)

    if direct:
        return direct

    for key in ("intrinsics","left_cam","left"):
        sub = cam.get(key)

        if isinstance(sub, dict):
            found = from_flat(sub)

            if found:
                return found

    for key in ("camera_matrix", "K", "intrinsic_matrix"):
        m = cam.get(key)

        if (m and len(m) == 3 and len(m[0]) == 3):
            return {
                "fx": float(m[0][0]),
                "fy": float(m[1][1]),
                "cx": float(m[0][2]),
                "cy": float(m[1][2])
            }

    return None


def zed_intrinsics(camera):
    """
    Intrinsics reported by ZED for the LEFT camera at the
    active resolution.
    """

    calib = (camera.get_camera_information().camera_configuration.calibration_parameters)
    left = calib.left_cam

    return {
        "fx": float(left.fx),
        "fy": float(left.fy),
        "cx": float(left.cx),
        "cy": float(left.cy)
    }


def unproject(px, py, z_m, intr):
    """
    Pixel + ZED optical-axis depth -> metric ZED camera coordinates.

    ZED camera coordinates:
        +X = right
        +Y = down
        +Z = forward
    """

    if z_m is None:
        return None, None, None

    if not np.isfinite(z_m) or z_m <= 0:
        return None, None, None

    x = ((px - intr["cx"]) * z_m / intr["fx"])
    y = ((py - intr["cy"]) * z_m / intr["fy"])

    return x, y, z_m


def zed_to_unreal(zed_xyz):
    """
    ZED camera coordinates -> Unreal coordinates.

    ZED:
        X = right
        Y = down
        Z = forward

    Unreal:
        X = forward
        Y = right
        Z = up
    """

    return np.array(
        [
            zed_xyz[2],
            zed_xyz[0],
            -zed_xyz[1],
        ],
        dtype=np.float32
    )


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
    ZED acquisition thread.
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
            [
                [
                    float(p[0]),
                    float(p[1]),
                ]
                for p in body.bounding_box_2d
            ],
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
                valid = (body.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
            except Exception:
                pass

            if not valid:
                continue

            roi = self.bbox_from_body(
                body,
                width,
                height,
                self.roi_padding
            )

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
            and (frame_id - self.last_detection_frame <= self.roi_hold)
        ):
            pass

        else:
            self.last_roi = None

        with self.state.frame_lock:
            self.state.roi = self.last_roi
            self.state.roi_frame_id = self.last_detection_frame

        self._roi_n += 1

        if self._roi_n >= 30:
            elapsed = (time.monotonic() - self._roi_t0)

            self.state.roi_fps = (
                self._roi_n / elapsed
                if elapsed > 0
                else 0.0
            )

            self._roi_n = 0
            self._roi_t0 = time.monotonic()

    def run(self):
        n = 0
        t0 = time.monotonic()

        while not self.state.stop.is_set():

            if (self.camera.grab(self.runtime) != sl.ERROR_CODE.SUCCESS):
                continue

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
                    "frame_id": frame_id
                }

                self.state.frame_id += 1

            self.state.new_frame.set()

            n += 1

            if n >= 30:
                elapsed = (time.monotonic() - t0)

                self.state.capture_fps = (
                    n / elapsed
                    if elapsed > 0
                    else 0.0
                )

                n = 0
                t0 = time.monotonic()


class _TasksResultShim:
    """
    Makes MediaPipe Tasks results compatible with
    the legacy result interface used by the program.
    """

    def __init__(self, task_result):

        if not task_result.hand_landmarks:
            self.multi_hand_landmarks = None
            self.multi_hand_world_landmarks = None
            self.multi_handedness = None
            return

        self.multi_hand_landmarks = []

        for lm_list in task_result.hand_landmarks:
            hand = type("Hand",(),{})()

            hand.landmark = lm_list

            self.multi_hand_landmarks.append(hand)

        self.multi_hand_world_landmarks = []

        for lm_list in (task_result.hand_world_landmarks):
            hand = type("WorldHand",(),{})()

            hand.landmark = lm_list

            self.multi_hand_world_landmarks.append(hand)

        self.multi_handedness = []

        for cat_list in task_result.handedness:
            h = type("Handedness",(),{})()

            classification = type("Classification",(),{})()

            classification.label = cat_list[0].category_name

            h.classification = [classification]

            self.multi_handedness.append(h)


class TrackingThread(threading.Thread):
    """
    MediaPipe tracking thread.

    Only the newest frame is processed.
    """

    def __init__(
        self,
        state,
        width,
        max_hands,
        det_conf,
        track_conf,
        complexity,
        engine="legacy",
        delegate="cpu",
        model_path=None
    ):
        super().__init__(daemon=True)

        self.state = state
        self.width = width
        self.engine = engine

        self.last_frame_id = -1
        self._start = time.monotonic()

        if engine == "legacy":

            self.hands = (
                mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    model_complexity=complexity,
                    min_detection_confidence=det_conf,
                    min_tracking_confidence=track_conf
                )
            )

        else:
            from mediapipe.tasks.python import (BaseOptions)
            from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions, RunningMode)

            base_options = BaseOptions(
                model_asset_path=model_path,
                delegate=(
                    BaseOptions.Delegate.GPU
                    if delegate == "gpu"
                    else BaseOptions.Delegate.CPU
                )
            )

            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_tracking_confidence=track_conf
            )

            self.landmarker = HandLandmarker.create_from_options(options)

    def _infer(self, rgb):

        if self.engine == "legacy":

            rgb.flags.writeable = False

            results = self.hands.process(rgb)

            rgb.flags.writeable = True

            return results

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        timestamp_ms = int((time.monotonic() - self._start) * 1000)

        task_result = self.landmarker.detect_for_video(image, timestamp_ms)

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

                if (frame_id == self.last_frame_id):
                    continue

                self.last_frame_id = frame_id

                h, w = frame.shape[:2]

                if roi is not None:

                    x0, y0, x1, y1 = roi

                    x0 = max(0, min(w - 1, x0))
                    y0 = max(0, min(h - 1, y0))

                    x1 = max(x0 + 1, min(w, x1))
                    y1 = max(y0 + 1, min(h, y1))

                    inp = frame[y0:y1, x0:x1]

                else:
                    x0 = 0
                    y0 = 0
                    x1 = w
                    y1 = h

                    inp = frame

                roi_h, roi_w = inp.shape[:2]

                if (self.width and roi_w > self.width):

                    scale = (self.width / float(roi_w))

                    inp = cv2.resize(
                        inp,
                        (
                            self.width,
                            max(1, int(roi_h * scale))
                        ),
                        interpolation=cv2.INTER_AREA
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
                        "roi_width": roi_w,
                        "roi_height": roi_h,
                    }

                n += 1

                if n >= 30:
                    elapsed = (time.monotonic() - t0)

                    self.state.tracking_fps = (
                        n / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    n = 0
                    t0 = time.monotonic()

        finally:

            if self.engine == "legacy":
                self.hands.close()
            else:
                self.landmarker.close()


def depth_at(depth, x, y):
    """
    Read ZED depth at a pixel.
    Uses a 5x5 median fallback when the exact pixel is invalid.
    """
    h, w = depth.shape[:2]

    x = int(round(x))
    y = int(round(y))

    if not (0 <= x < w and 0 <= y < h):
        return None

    z = float(depth[y, x])

    if (np.isfinite(z) and z > 0):
        return z

    values = []

    for dy in range(-2, 3):
        for dx in range(-2, 3):

            xx = x + dx
            yy = y + dy

            if not (0 <= xx < w and 0 <= yy < h):
                continue

            zz = float(depth[yy, xx])

            if (np.isfinite(zz) and zz > 0):
                values.append(zz)

    if not values:
        return None

    return float(np.median(values))


def landmark_to_original_pixel(lm,result):
    """
    Map MediaPipe normalized image coordinates
    back to native ZED pixels.
    """

    x0, y0, x1, y1 = result["roi"]

    roi_w = result["roi_width"]
    roi_h = result["roi_height"]

    x = (x0 + float(lm.x) * roi_w)
    y = (y0 + float(lm.y) * roi_h)

    return x, y


def reconstruct_hand_3d(
    hand,
    result,
    depth,
    intr,
    hand_scale
):
    """
    Reconstruct a hand using:

        absolute position:
            ZED wrist depth

        relative shape:
            MediaPipe's 21 landmarks

    Returns 21 points in ZED camera coordinates, in meters.
    """

    if hand is None:
        return None

    landmarks = hand.landmark

    if len(landmarks) != 21:
        return None

    wrist = landmarks[0]

    # ------------------------------------------------------------
    # Absolute wrist position from ZED.
    # ------------------------------------------------------------

    wrist_px, wrist_py = landmark_to_original_pixel(wrist, result)

    wrist_depth = depth_at(depth, wrist_px, wrist_py,)

    if wrist_depth is None:
        return None

    wx, wy, wz = unproject(wrist_px, wrist_py, wrist_depth, intr)

    if wx is None:
        return None

    wrist_zed = np.array( [wx, wy, wz], dtype=np.float32)

    # ------------------------------------------------------------
    # MediaPipe hand geometry.
    #
    # x/y are normalized image coordinates.
    # z is relative landmark depth.
    #
    # Everything is made relative to the wrist.
    # ------------------------------------------------------------

    roi_w = float(result["roi_width"])
    roi_h = float(result["roi_height"])

    # Use the larger dimension as the common normalized
    # hand-space scale.
    image_dimension = max(roi_w, roi_h)

    reconstructed = []

    for lm in landmarks:

        dx = (float(lm.x - wrist.x)* roi_w)
        dy = (float(lm.y - wrist.y) * roi_h)
        dz = float(lm.z - wrist.z) * image_dimension

        # Convert the MediaPipe relative hand into a consistent
        # local metric shape.
        #
        # NOTE:
        # hand_scale global. Don't allow landmarks to use ZED depths besides wrist.
        local = np.array([dx, dy, dz], dtype=np.float32)

        # Normalize pixel-scale coordinates before applying the
        # configurable global scale.
        local /= image_dimension

        # Convert the normalized hand shape into a relative
        # ZED-camera-frame offset.
        #
        # ZED:
        #   X = right
        #   Y = down
        #   Z = forward
        #
        # MediaPipe x/y/z are treated as the local hand shape.
        offset = ( local * float(hand_scale))

        reconstructed.append(wrist_zed + offset)

    return reconstructed


def points_to_finger_dict(points):
    """
    Convert reconstructed ZED points into Unreal coordinates
    and group them into the existing payload format.

    Output units: centimeters.
    """

    if points is None:
        return None

    out = {}

    for finger, indices in (FINGER_GROUPS.items()):
        finger_points = []

        for idx in indices:

            if (idx >= len(points) or points[idx] is None):
                finger_points.append([None, None, None])
                continue

            ue = zed_to_unreal(points[idx])

            finger_points.append(
                [
                    float(ue[0] * 100.0),
                    float(ue[1] * 100.0),
                    float(ue[2] * 100.0)
                ]
            )

        out[finger] = finger_points

    return out


def build_finger_dict(
    hand,
    result,
    depth,
    intr,
    hand_scale
):
    """
    Build one complete hand.
    Returns None if hand can't be reconstructed.
    """

    if hand is None:
        return None

    points = reconstruct_hand_3d(hand, result, depth, intr, hand_scale)

    if points is None:
        return None

    return points_to_finger_dict(points)


def send_message(payload: dict):
    """
    Replace this with Unreal message transport.
    UDP / OSC / WebSocket / TCP can be implemented here.
    """
    pass


def output_landmarks(
    results,
    depth,
    result,
    intr,
    hand_scale
):
    """
    Output ONLY successfully tracked/reconstructed hands.

    Good tracking:
        {'hands': {
            'Left': {...},
            'Right': {...}
        }}
    One hand:
        {'hands': {
            'Left': {...}
        }}
    No valid tracking:
        nothing is printed or sent.
    """

    if (
        results is None
        or not results.multi_hand_landmarks
    ):
        return

    hands_payload = {}

    handedness = (
        results.multi_handedness
        or []
    )

    for i, hand in enumerate( results.multi_hand_landmarks):

        label = "Unknown"
        
        # Handedness labeling
        if i < len(handedness):

            raw = handedness[i].classification[0].label

            # Overhead-camera inverts handedness, mediapipe default expects selfie
            if raw == "Left":
                label = "Right"
            elif raw == "Right":
                label = "Left"
            else:
                label = raw

        # --------------------------------------------------------
        # Reconstruct the actual hand.
        # --------------------------------------------------------

        hand_payload = build_finger_dict(hand, result, depth, intr, hand_scale)

        # --------------------------------------------------------
        # A detected hand with invalid ZED wrist depth should NOT
        # become: "Left": None
        #
        # Omit from outgoing messages
        # --------------------------------------------------------

        if hand_payload is None:
            continue

        hands_payload[label] = hand_payload

    if not hands_payload:
        return

    message_payload = {"hands": hands_payload}
    print(message_payload)
    send_message(message_payload)


def draw_results(display, result):
    """
    Draw MediaPipe 2D skeleton.
    """
    results = result["results"]

    if (results is None or not results.multi_hand_landmarks):
        return

    x0, y0, x1, y1 = result["roi"]

    roi_w = result["roi_width"]
    roi_h = result["roi_height"]

    for hand in (results.multi_hand_landmarks):

        points = []

        for lm in hand.landmark:

            x = int(x0 + lm.x * roi_w)
            y = int(y0 + lm.y * roi_h)

            points.append((x, y))

        for a, b in (mp.solutions.hands.HAND_CONNECTIONS):

            if (0 <= a < len(points) and 0 <= b < len(points)):
                cv2.line(
                    display,
                    points[a],
                    points[b],
                    (0, 255, 0),
                    2
                )

        for x, y in points:
            cv2.circle(
                display,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

    cv2.rectangle(
        display,
        (int(x0), int(y0)),
        (int(x1), int(y1)),
        (255, 0, 0),
        2
    )


def main():

    a = args()

    serial, cam_info = load_calibration(a.calib)

    cv2.setNumThreads(a.cv_threads)

    print(
        "[info] "
        f"cv2.setNumThreads({a.cv_threads})"
    )

    resolutions = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA
    }

    init = sl.InitParameters()

    init.set_from_serial_number(serial)

    init.camera_resolution = resolutions[a.resolution]

    init.camera_fps = a.fps

    init.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT

    init.coordinate_units = sl.UNIT.METER

    camera = sl.Camera()

    status = camera.open(init)

    if (status != sl.ERROR_CODE.SUCCESS):
        raise RuntimeError(f"ZED X Mini open failed: {status}")

    # Fix exposure to optimize
    camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    camera.set_camera_settings( sl.VIDEO_SETTINGS.EXPOSURE, 40)
    camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 10)

    info = camera.get_camera_information()

    cfg = info.camera_configuration

    print(
        f"[info] Opened {info.camera_model}: "
        f"{cfg.resolution.width}x"
        f"{cfg.resolution.height} "
        f"@ {cfg.fps} FPS"
    )

    print(f"[info] Serial: {serial}")
    print("[info] ZED retrieval memory: GPU")
    
    print(
        "[info] Body-ROI cropping: "
        f"{'ON' if a.use_roi else 'OFF'}"
    )

    print(
        "[info] MediaPipe engine: "
        f"{a.engine}"
    )

    print(
        "[info] Hand reconstruction: "
        "ZED wrist depth + MediaPipe "
        "relative hand geometry"
    )

    print(
        "[info] Hand scale: "
        f"{a.hand_scale}"
    )

    print("[info] Unreal mapping: X=ZED.Z, Y=ZED.X, Z=-ZED.Y")

    intr = try_json_intrinsics(cam_info)

    if intr is not None:

        print(
            "[info] Intrinsics source: "
            "calibration JSON "
            f"fx={intr['fx']:.2f} "
            f"fy={intr['fy']:.2f} "
            f"cx={intr['cx']:.2f} "
            f"cy={intr['cy']:.2f}"
        )

    else:
        intr = zed_intrinsics(camera)

        print("[warn] Could not find fx/fy/cx/cy in calibration JSON.")

        print("[info] Using ZED SDK intrinsics.")

        print(
            f"[info] fx={intr['fx']:.2f} "
            f"fy={intr['fy']:.2f} "
            f"cx={intr['cx']:.2f} "
            f"cy={intr['cy']:.2f}"
        )

    if a.use_roi:
        body_params = sl.BodyTrackingParameters()

        body_params.enable_tracking = False
        body_params.enable_body_fitting = False
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_params.body_format = sl.BODY_FORMAT.BODY_18

        status = camera.enable_body_tracking(body_params)

        if (status != sl.ERROR_CODE.SUCCESS):
            raise RuntimeError(
                "ZED body tracking enable failed: "
                f"{status}"
            )

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
        model_path=a.model_path
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

            # Make calibration info available to output routine.
            if result is not None:
                result["intrinsics"] = intr

            if not a.no_display:
                display = current["image"].copy()

                if result is not None:
                    mp_results = result["results"]

                    if (mp_results and mp_results.multi_hand_landmarks):

                        draw_results(display, result)

                        now = time.monotonic()

                        if ( now - last_print >= 1.0):

                            output_landmarks(
                                mp_results,
                                result["depth"],
                                result,
                                intr,
                                a.hand_scale
                            )

                            last_print = now

                # FPS Info
                """cv2.putText(
                    display,
                    (
                        f"Capture: "
                        f"{state.capture_fps:.1f} FPS"
                    ),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    (
                        f"MediaPipe: "
                        f"{state.tracking_fps:.1f} FPS"
                    ),
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                if a.use_roi:

                    cv2.putText(
                        display,
                        (
                            f"Body ROI: "
                            f"{state.roi_fps:.1f} FPS"
                        ),
                        (15, 86),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                """
                cv2.imshow("ZED X Mini + MediaPipe Hands", display)

                if (cv2.waitKey(1) & 0xFF == ord("q")):
                    state.stop.set()
                    break

            else:
                if result is not None:
                    mp_results = result["results"]

                    now = time.monotonic()

                    if (now - last_print >= 1.0):

                        output_landmarks(
                            mp_results,
                            result["depth"],
                            result,
                            intr,
                            a.hand_scale
                        )

                        print(
                            f"Capture: "
                            f"{state.capture_fps:.1f} FPS"
                        )

                        print(
                            f"MediaPipe: "
                            f"{state.tracking_fps:.1f} FPS"
                        )

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


if __name__ == "__main__":
    main()
