#aruco_board.py

import cv2
import numpy as np

from scipy.spatial.transform import Rotation


# ============================================================
# BOARD CONFIGURATION
# ============================================================

ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100

BOARD_COLUMNS = 5
BOARD_ROWS = 7

MARKER_SIZE = 0.02861          # 28.61 mm
MARKER_SEPARATION = 0.03828    # 38.28 mm edge-to-edge gap

SQUARE_SIZE = (MARKER_SIZE + MARKER_SEPARATION)

CV_TO_UE = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0,-1, 0],
])


# ============================================================
# BOARD
# ============================================================

class ArucoBoard:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTIONARY
        )

        self.marker_size = MARKER_SIZE
        self.square_size = SQUARE_SIZE

        self.columns = BOARD_COLUMNS
        self.rows = BOARD_ROWS

        self.board = cv2.aruco.CharucoBoard(
            (
                self.columns,
                self.rows
            ),
            self.square_size,
            self.marker_size,
            self.dictionary
        )

        self.detector = self._create_detector()

    # --------------------------------------------------------

    def _create_detector(self):
        parameters = cv2.aruco.DetectorParameters()

        parameters.cornerRefinementMethod = (cv2.aruco.CORNER_REFINE_SUBPIX)

        return cv2.aruco.ArucoDetector(self.dictionary, parameters)

    # --------------------------------------------------------

    def _create_object_points(self):
        """
        Creates the known 3D positions of all board marker corners.

        Board lies on Z = 0.

        Origin:
            top-left corner of marker 0
        """

        points = {}

        for row in range(self.rows):

            for col in range(self.columns):

                marker_id = row * self.columns + col

                x = col * self.marker_step
                y = row * self.marker_step

                points[marker_id] = np.array(
                    [
                        [x, y, 0.0],
                        [x + self.marker_size, y, 0.0],
                        [x + self.marker_size, y + self.marker_size, 0.0],
                        [x, y + self.marker_size, 0.0],
                    ],
                    dtype=np.float32
                )

        return points

    # --------------------------------------------------------

    def detect(self, image):

        #Detect ArUco markers.

        corners, ids, rejected = self.detector.detectMarkers(image)

        return corners, ids

    # --------------------------------------------------------
    
    def charuco_pose_to_unreal(self, rvec, tvec):
        # Board -> cam
        R_cv, _ = cv2.Rodrigues(rvec)
        
        # Cam -> board
        R_cam = R_cv.T
        t_cam = -R_cam @ tvec.reshape(3,1)
        
        # OpenCV -> Unreal
        R_ue = CV_TO_UE @ R_cam @ CV_TO_UE.T
        t_ue = CV_TO_UE @ t_cam
        
        #meters -> centimeters
        t_ue *= 100.0
        
        # Unreal Euler
        rot = Rotation.from_matrix(R_ue).as_euler('xyz', degrees=True)
        
        print(t_ue.flatten())
        print(rot)
        
        return

    # --------------------------------------------------------

    def estimate_pose(
        self,
        corners,
        ids,
        frame,
        camera_matrix,
        distortion
    ):
        
       #Estimate board pose relative to camera.
        if ids is None or len(ids) == 0:
            return None

        retval, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                image=frame,
                board=self.board,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion
            )
        )

        if (charuco_ids is None or retval < 4):
            return None

        success, rvec, tvec = (
            cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                camera_matrix,
                distortion,
                None,
                None
            )
        )

        if not success:
            return None

        rotation_board_to_camera, _ = (cv2.Rodrigues(rvec))

        camera_position = (-rotation_board_to_camera.T @ tvec).reshape(3)

        camera_rotation = (rotation_board_to_camera.T)

        projected, _ = cv2.projectPoints(
            self.board.getChessboardCorners()[
                charuco_ids.flatten()
            ],
            rvec,
            tvec,
            camera_matrix,
            distortion
        )

        projected = projected.reshape(-1, 2)

        observed = charuco_corners.reshape(-1, 2)

        reprojection_error = float(
            np.mean(
                np.linalg.norm(
                    projected - observed,
                    axis=1
                )
            )
        )

        self.charuco_pose_to_unreal(rvec, tvec)

        return {

            "rvec":
                rvec.reshape(3),

            "tvec":
                tvec.reshape(3),

            "camera_position":
                camera_position,

            "camera_rotation":
                camera_rotation,

            "reprojection_error":
                reprojection_error,

            "marker_ids":
                ids.flatten().tolist(),

            "marker_count":
                len(ids),

            "charuco_corner_count":
                len(charuco_ids)
        }
    
    # --------------------------------------------------------

    def draw(
        self,
        image,
        corners,
        ids,
        pose,
        camera_matrix,
        distortion
    ):
        """
        Draw detected markers and board coordinate axes.
        """

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(image,corners,ids)

        retval, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                image,
                self.board,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion
            )
        )
        
        '''
        print(
            f"Markers: {len(ids)}, "
            f"Charuco corners: {0 if charuco_ids is None else len(charuco_ids)}"
        )
        '''

        if (charuco_ids is not None and len(charuco_ids) > 0):

            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

        if pose is None:
            return image

        cv2.drawFrameAxes(
            image,
            camera_matrix,
            distortion,
            pose["rvec"].reshape(3, 1),
            pose["tvec"].reshape(3, 1),
            self.square_size,
            2
        )

        return image
        
