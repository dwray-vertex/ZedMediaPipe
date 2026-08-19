#stereo_camera.py

import cv2
import pyzed.sl as sl
import numpy as np


from camera import Camera



class StereoCamera(Camera):


    def __init__(self, name, serial):

        super().__init__(name)

        self.serial = serial

        self.zed = sl.Camera()

        self.image = sl.Mat()

        self.runtime = sl.RuntimeParameters()



    def open(self):


        init = sl.InitParameters()


        init.set_from_serial_number(self.serial)


        init.camera_resolution = (sl.RESOLUTION.HD1200)


        init.camera_fps = 30


        result = self.zed.open(init)


        if result != sl.ERROR_CODE.SUCCESS:

            raise RuntimeError(f"Failed opening {self.name}")



        info = self.zed.get_camera_information()


        calibration = (
            info.camera_configuration
            .calibration_parameters
            .left_cam
        )



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



        self.distortion = np.array(
            calibration.disto[:5],
            dtype=np.float64
        )



    def grab(self):


        if (
            self.zed.grab(self.runtime)
            != sl.ERROR_CODE.SUCCESS
        ):

            return None



        self.zed.retrieve_image(
            self.image,
            sl.VIEW.LEFT
        )


        frame = self.image.get_data()


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )


        return frame



    def close(self):

        self.zed.close()
        
