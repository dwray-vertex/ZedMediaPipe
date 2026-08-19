#config.py

import cv2
import numpy as np

class Camera:


    def __init__(self, name):

        self.name = name

        self.K = None

        self.distortion = None



    def open(self):

        raise NotImplementedError



    def grab(self):

        raise NotImplementedError



    def close(self):

        raise NotImplementedError



    def get_intrinsics(self):

        return self.K, self.distortion
