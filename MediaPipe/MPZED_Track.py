import argparse
import numpy as np
import pyzed.sl as sl
import cv2
import json
import mediapipe as mp
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional
 
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", default="/home/user/Documents/Ultraleap_ChArUco_Python_Calibration/camera_calibration.json", 
                    help="Path to calibration JSON")
    p.add_argument("--resolution", default="HD1080",
                    choices=["HD2K", "HD1080", "HD720", "VGA"],
                    help="ZED capture resolution (default: HD720, good balance for real-time)")
    p.add_argument("--fps", type=int, default=30, help="Requested ZED camera FPS")
    p.add_argument("--max-hands", type=int, default=2, help="Max hands for MediaPipe to track")
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    p.add_argument("--detect-width", type=int, default=480,
                    help="Downscale width fed to MediaPipe (0 = no downscale)")
    p.add_argument("--no-display", action="store_true",
                    help="Run headless (no cv2.imshow window) -- For test over SSH")
    p.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (useful for headless test)")
    return p.parse_args()


@dataclass
class CameraCalib:
    serial_number: int
    name: str
    R: np.ndarray              # 3x3, this camera -> main/mini frame
    T: np.ndarray              # 3x1 meters, this camera -> main/mini frame
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
 
 
def load_calibration(path: str):
    """Returns (dict[serial_number -> CameraCalib])."""
    with open(path, "r") as f:
        raw = json.load(f)
 
    calibs = {}
    for name, cam in raw["cameras"].items():
        transform = np.array(cam["transform_camera_to_main"], dtype=np.float64)
        R = transform[:3, :3]
        T = transform[:3, 3].reshape(3, 1)
        calibs[cam["serial"]] = CameraCalib(
            serial_number=cam["serial"], name=name, R=R, T=T,
        )
    return calibs

 
class CameraWorker(threading.Thread):
    def __init__(self, sl, cv2, mp_hands, serial_number: int, calib: CameraCalib, 
                 min_det_conf: float, min_track_conf: float, detect_width: int):
        super().__init__(daemon=True)
        self.sl = sl
        self.cv2 = cv2
        self.calib = calib
        self.serial_number = serial_number
        self.detect_width = detect_width
 
        isMain = (calib.name=="main")
        init_params = sl.InitParameters() if isMain else sl.InitParametersOne()     
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.input.set_from_serial_number(serial_number)
 
        if isMain:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        self.cam = sl.Camera() if isMain else sl.CameraOne()
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open camera SN={serial_number} ({calib.name}): {status}")
 
        # Pull real intrinsics from the SDK now that the camera is open.
        calib_params = self.cam.get_camera_information().camera_configuration.calibration_parameters

        if isMain:
            left_cam = calib_params.left_cam
            calib.fx, calib.fy, calib.cx, calib.cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
        else:
            calib.fx, calib.fy, calib.cx, calib.cy = calib_params.fx, calib_params.fy, calib_params.cx, calib_params.cy
 
        self.runtime_params = sl.RuntimeParameters() if isMain else None
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat() if isMain else None
 
        # --------------------------- MediaPipe Hand Detection --------------------------- 
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
            model_complexity=1
        )
 
        self._lock = threading.Lock()
        self._latest = {"frame_bgr": None, "results": None, "score": 0.0,
                         "frame_w": None, "frame_h": None, "ts": 0.0}
        self._stop_flag = threading.Event()
        self._fps_count = 0
        self._fps_window_start = time.time()
        self.last_fps = 0.0
 
    def run(self):
        cv2 = self.cv2
        isMain = (self.calib.name=="main")
        while not self._stop_flag.is_set():
            if self.runtime_params is not None:
                if self.cam.grab(self.runtime_params) != self.sl.ERROR_CODE.SUCCESS:
                    continue
 

            if isMain:
                self.cam.retrieve_image(self.image_mat, self.sl.VIEW.LEFT)
            else:
                self.cam.retrieve_image(self.image_mat)
            frame_bgra = self.image_mat.get_data()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            h, w = frame_bgr.shape[:2]
 
            if self.depth_mat is not None:
                self.cam.retrieve_measure(self.depth_mat, self.sl.MEASURE.DEPTH)
 
            # Downscale for detection only; landmarks are normalized [0,1] so
            # they map back to full-res pixels with no extra scaling math.
            if self.detect_width and w > self.detect_width:
                scale = self.detect_width / w
                small = cv2.resize(frame_bgr, (self.detect_width, int(h * scale)))
            else:
                small = frame_bgr
            
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
 
            results = self.hands.process(small_rgb)
            score = 0.0
            if results.multi_handedness:
                score = max(c.classification[0].score for c in results.multi_handedness)
 
            with self._lock:
                self._latest = {"frame_bgr": frame_bgr, "results": results, "score": score,
                                 "frame_w": w, "frame_h": h, "ts": time.time()}
 
            self._fps_count += 1
            if self._fps_count >= 30:
                elapsed = time.time() - self._fps_window_start
                self.last_fps = self._fps_count / elapsed if elapsed > 0 else 0.0
                self._fps_count = 0
                self._fps_window_start = time.time()
 
    def get_latest(self):
        with self._lock:
            return dict(self._latest)
 
    def get_depth_at(self, px: int, py: int) -> Optional[float]:
        if self.depth_mat is None:
            return None
        h, w = self.depth_mat.get_height(), self.depth_mat.get_width()
        if not (0 <= px < w and 0 <= py < h):
            return None
        z = self.depth_mat.get_data()[py, px]
        return float(z) if np.isfinite(z) and z > 0 else None
 
    def stop(self):
        self._stop_flag.set()
 
    def close(self):
        self.hands.close()
        self.cam.close()


def landmark_to_mini_frame(lm_x_px: float, lm_y_px: float, depth_m: float,
                            calib: CameraCalib) -> np.ndarray:
    x_cam = (lm_x_px - calib.cx) * depth_m / calib.fx
    y_cam = (lm_y_px - calib.cy) * depth_m / calib.fy
    z_cam = depth_m
    p_cam = np.array([[x_cam], [y_cam], [z_cam]])
    p_mini = calib.R @ p_cam + calib.T
    return p_mini.flatten()


 
def main():
    args = parse_args()
    #args.no_display = True
    
    # Read calibration file
    calibs = load_calibration(args.calib)
    
    mini_serials = [sn for sn, c in calibs.items() if c.name == "main"]
    mini_sn = mini_serials[0]
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    workers = {}
    try:
        for sn, calib in calibs.items():
            print(f"[info] Opening SN={sn} name={calib.name} ...")

            w = CameraWorker(
                sl, cv2, mp_hands, sn, calib, args.min_detection_confidence,
                args.min_tracking_confidence, args.detect_width
            )
            workers[sn] = w
            w.start()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        for w in workers.values():
            w.stop(); w.close()
        sys.exit(1)
 
    last_mini_wrist_depth = None
    frame_count = 0

    window_name = "MediaPipe Zed Hands"
 
    print("[info] Starting capture loop. Press 'q' in the display window to quit or Ctrl+C from terminal.")
    time.sleep(1.0)
 
    try:
        while True:
            if args.max_frames is not None and frame_count >= args.max_frames:
                break
 
            latest = {sn: w.get_latest() for sn, w in workers.items()}
            scores = {sn: d["score"] for sn, d in latest.items()}
            best_sn = max(scores, key=scores.get)
            best_score = scores[best_sn]
 
            if best_score == 0.0 or latest[best_sn]["results"] is None:
                if not args.no_display and latest.get(mini_sn, {}).get("frame_bgr") is not None:
                    cv2.imshow(window_name, latest[mini_sn]["frame_bgr"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_count += 1
                continue
 
            best_calib = calibs[best_sn]
            best_data = latest[best_sn]
            best_results = best_data["results"]
            w, h = best_data["frame_w"], best_data["frame_h"]
            hand_landmarks = best_results.multi_hand_landmarks[0]
            handedness_label = best_results.multi_handedness[0].classification[0].label
 
            if best_calib.name == "main":
                wrist_lm = hand_landmarks.landmark[0]
                wrist_px, wrist_py = int(wrist_lm.x * w), int(wrist_lm.y * h)
                d = workers[best_sn].get_depth_at(wrist_px, wrist_py)
                if d is not None:
                    last_mini_wrist_depth = d
                depth_source = "measured (Mini)"
            else:
                depth_source = "APPROXIMATED (last known Mini depth)" if last_mini_wrist_depth else "UNAVAILABLE"
 
            depth_m = last_mini_wrist_depth
            points_mini_frame = []
            if depth_m is not None:
                for lm in hand_landmarks.landmark:
                    px, py = lm.x * w, lm.y * h
                    points_mini_frame.append(
                        landmark_to_mini_frame(px, py, depth_m, best_calib)
                    )
 
            if frame_count % 30 == 0:
                fps_report = {sn: round(w_.last_fps, 1) for sn, w_ in workers.items()}
                print(f"[frame {frame_count}] best=SN{best_sn}({best_calib.name}) "
                      f"hand={handedness_label} score={best_score:.2f}")
 
            if not args.no_display:
                display = best_data["frame_bgr"].copy()
                mp_drawing.draw_landmarks(
                    display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                cv2.putText(display, f"{best_calib.name} score={best_score:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
            frame_count += 1
 
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
 
    finally:
        for w in workers.values():
            w.stop()
        for w in workers.values():
            w.join(timeout=2.0)
            w.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"[info] Done. Processed {frame_count} main-loop iterations.")
        
        
if __name__ == "__main__":
    main()

