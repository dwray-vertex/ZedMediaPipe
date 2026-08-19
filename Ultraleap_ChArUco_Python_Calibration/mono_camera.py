#mono_camera.py

import cv2
import numpy as np
import pyzed.sl as sl

from camera import Camera



class MonoCamera(Camera):


    def __init__(self, name, serial):


        super().__init__(name)

        self.serial = serial

        self.zed = sl.CameraOne()

        self.image = sl.Mat()



    def open(self):

        init = sl.InitParametersOne()

        init.set_from_serial_number(self.serial)

        init.camera_resolution = sl.RESOLUTION.HD1200
        init.camera_fps = 30

        result = self.zed.open(init)

        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed opening {self.name}")

        info = self.zed.get_camera_information()

        # Monocular camera calibration
        calibration = (info.camera_configuration.calibration_parameters)

        self.K = np.array(
            [
                [calibration.fx, 0, calibration.cx],
                [0, calibration.fy, calibration.cy],
                [0, 0, 1]
            ],
            dtype=np.float64
        )

        self.distortion = np.array(calibration.disto[:5], dtype=np.float64)


        self.K = np.array(
            [
                [
                    calibration.fx,
                    0,
                    calibration.cx
                ],

                [
                    0,
                    calibration.fy,
                    calibration.cy
                ],

                [
                    0,
                    0,
                    1
                ]
            ],
            dtype=np.float64
        )



        self.distortion = np.array(calibration.disto[:5], dtype=np.float64)



    def grab(self):

        err = self.zed.grab()

        if err != sl.ERROR_CODE.SUCCESS:
            return None 
        
        self.zed.retrieve_image(self.image)


        frame = self.image.get_data()


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )


        return frame



    def close(self):

        if self.zed:

            self.zed.close()
