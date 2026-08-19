
# calibrate.py

import cv2
import json
import time
import numpy as np
import pyzed.sl as sl

from aruco_board import ArucoBoard
from mono_camera import MonoCamera
from stereo_camera import StereoCamera


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERAS = [
    {
        "name": "main",
        "type": "stereo",
        "serial": 59707335
    },
    {
        "name": "right",
        "type": "mono",
        "serial": 309357335
    },
    {
        "name": "left",
        "type": "mono",
        "serial": 300533151
    }
]

REFERENCE_CAMERA = "main"


# ============================================================
# CALIBRATION SETTINGS
# ============================================================

MIN_MARKERS = 2
MIN_VALID_FRAMES = 20
MAX_REPROJECTION_ERROR = 3.0

OUTPUT_FILE = "camera_calibration.json"

SHOW_WINDOWS = True


# ============================================================
# TRANSFORM UTILITIES
# ============================================================

def make_transform(rotation, translation):

    transform = np.eye(
        4,
        dtype=np.float64
    )

    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def invert_transform(transform):

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(
        4,
        dtype=np.float64
    )

    inverse[:3, :3] = rotation.T

    inverse[:3, 3] = (
        -rotation.T @ translation
    )

    return inverse


# ============================================================
# POSE ACCUMULATOR
# ============================================================

class PoseAccumulator:

    def __init__(self):

        self.positions = []
        self.rotations = []

    def add(self, pose):

        self.positions.append(pose["camera_position"])

        self.rotations.append(pose["camera_rotation"])

    def count(self):

        return len(self.positions)

    def position(self):

        return np.mean(
            np.asarray(self.positions),
            axis=0
        )

    def rotation(self):

        matrices = np.asarray(self.rotations)

        average = np.mean(matrices, axis=0)

        U, _, Vt = np.linalg.svd(average)

        rotation = U @ Vt

        if np.linalg.det(rotation) < 0:

            U[:, -1] *= -1

            rotation = U @ Vt

        return rotation

    def transform(self):

        return make_transform(
            self.rotation(),
            self.position()
        )


# ============================================================
# CAMERA CREATION
# ============================================================

def create_camera(config):

    if config["type"] == "mono":

        return MonoCamera(
            config["name"],
            config["serial"]
        )

    if config["type"] == "stereo":

        return StereoCamera(
            config["name"],
            config["serial"]
        )

    raise ValueError(
        f"Unknown camera type: "
        f"{config['type']}"
    )


# ============================================================
# OPEN CAMERAS
# ============================================================

def open_cameras():

    cameras = {}

    for config in CAMERAS:

        camera = create_camera(config)

        camera.open()

        cameras[config["name"]] = camera

    return cameras


# ============================================================
# CALIBRATION
# ============================================================

def calibrate():

    board = ArucoBoard()

    cameras = open_cameras()

    accumulators = {
        name:
            PoseAccumulator()
        for name in cameras
    }

    try:

        print()
        print(
            "============================================"
        )
        print(
            " ArUco / ZED Camera Calibration"
        )
        print(
            "============================================"
        )

        print(
            f"Reference camera: "
            f"{REFERENCE_CAMERA}"
        )

        print()

        while True:

            all_finished = True

            for name, camera in cameras.items():
            
                # Turn off auto exposure and reduce gain noise and exposure 
                # so whites arent so bright and contrast makes markers readable 

                camera.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
                
                camera.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 0)
                
                camera.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 15)

                frame = camera.grab()

                if frame is None:

                    all_finished = False

                    continue

                corners, ids = (board.detect(frame))

                pose = None

                if (ids is not None and len(ids) >= MIN_MARKERS):
                    
                    print(name)

                    pose = board.estimate_pose(corners, ids, frame, camera.K, camera.distortion)
                    
                    #print(pose)

                if pose is not None:

                    if (pose["reprojection_error"] <= MAX_REPROJECTION_ERROR):

                        accumulators[name].add(pose)

                if (accumulators[name].count()< MIN_VALID_FRAMES):

                    all_finished = False

                # --------------------------------------------
                # Display
                # --------------------------------------------

                if SHOW_WINDOWS:

                    display = frame.copy()

                    board.draw(display, corners, ids, pose, camera.K, camera.distortion)

                    count = (accumulators[name].count())

                    cv2.putText(
                        display,
                        (
                            f"{name}: "
                            f"{count}/"
                            f"{MIN_VALID_FRAMES}"
                        ),
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    if pose is not None:

                        position = (pose["camera_position"])

                        cv2.putText(
                            display,
                            (
                                f"X: "
                                f"{position[0]:+.3f}  "
                                f"Y: "
                                f"{position[1]:+.3f}  "
                                f"Z: "
                                f"{position[2]:+.3f}"
                            ),
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            display,
                            (
                                f"Markers: "
                                f"{pose['marker_count']}  "
                                f"Error: "
                                f"{pose['reprojection_error']:.2f}px"
                            ),
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                    cv2.imshow(name,display)

            # --------------------------------------------
            # Keyboard Escape before Calibration finish
            # --------------------------------------------

            if SHOW_WINDOWS:

                key = (cv2.waitKey(1)& 0xFF)

                # Esc
                if key == 27:

                    return

            if all_finished:

                print()
                print("All cameras localized.")

                break

            time.sleep(0.001)

        # ========================================================
        # BOARD-RELATIVE TRANSFORMS
        # ========================================================

        board_to_camera = {}

        for name in cameras:

            board_to_camera[name] = (
                accumulators[name].transform()
            )

        # ========================================================
        # MAIN CAMERA AS REFERENCE
        # ========================================================

        main_transform = (board_to_camera[REFERENCE_CAMERA])

        board_to_main = (invert_transform(main_transform))

        camera_to_main = {}

        for name in cameras:

            camera_to_main[name] = (
                board_to_main
                @ board_to_camera[name]
            )

        # ========================================================
        # BOARD-RELATIVE RESULTS
        # ========================================================

        print()
        print("============================================")
        #Multiply by 100 for Unreal values
        print(" CAMERA POSITIONS IN BOARD COORDINATES")
        print("============================================")

        for name in cameras:

            transform = (board_to_camera[name])

            position = (transform[:3, 3])

            print()
            print(name)

            print(
                f"  X: "
                f"{position[0]:+.6f} m"
            )

            print(
                f"  Y: "
                f"{position[1]:+.6f} m"
            )

            print(
                f"  Z: "
                f"{position[2]:+.6f} m"
            )

        # ========================================================
        # MAIN-RELATIVE RESULTS
        # ========================================================

        print()
        print("============================================")
        print(
            f" CAMERA TRANSFORMS RELATIVE TO "
            f"{REFERENCE_CAMERA}"
        )
        print("============================================")

        for name in cameras:

            transform = (camera_to_main[name])

            position = (transform[:3, 3])

            print()
            print(name)

            print(
                f"  X: "
                f"{position[0]:+.6f} m"
            )

            print(
                f"  Y: "
                f"{position[1]:+.6f} m"
            )

            print(
                f"  Z: "
                f"{position[2]:+.6f} m"
            )

            print("  Transform:")
            print(transform)

        # ========================================================
        # SAVE RESULTS
        # ========================================================

        output = {
            "reference_camera":
                REFERENCE_CAMERA,

            "cameras": {}
        }

        for config in CAMERAS:

            name = config["name"]

            board_transform = (board_to_camera[name])

            main_transform = (camera_to_main[name])

            output["cameras"][name] = {

                "serial":
                    config["serial"],

                "type":
                    config["type"],

                "samples":
                    accumulators[
                        name
                    ].count(),

                "position_board_m":
                    board_transform[
                        :3, 3
                    ].tolist(),

                "rotation_board":
                    board_transform[
                        :3, :3
                    ].tolist(),

                "transform_camera_to_main":
                    main_transform.tolist()
            }

        with open(
            OUTPUT_FILE,
            "w"
        ) as file:

            json.dump(
                output,
                file,
                indent=4
            )

        print()
        print(
            f"Saved calibration to "
            f"{OUTPUT_FILE}"
        )

    finally:

        for camera in cameras.values():

            camera.close()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    calibrate()


