#!/usr/bin/env python
# coding: utf-8
import math
from abc import abstractmethod, ABCMeta

import numpy as np

class Projector(metaclass=ABCMeta):

    @abstractmethod
    def in_simplex(self, w):
        pass

    @abstractmethod
    def project(self, center):
        pass


class ParamIn0to1Projector(Projector):
    """
    Projection operator to C = R^[0, 1]
    """
    def in_simplex(self, w):
        flag_0 = (w >= 0.0).all()
        flad_1 = (w <= 1.0).all()
        if flag_0 and flad_1:
            return True
        else:
            return False

    def project(self, center):
        """
        もしcenterがCに入ってなかったら，
        Cに対してcenterから垂線を引き，交点を返す
        """
        if self.in_simplex(center):
            return center
        else:
            center = np.where(center < 0.0, 0.0, center)
            center = np.where(center > 1.0, 1.0, center)
            return center

        # a_nxt = np.zeros(center.shape)
        #
        # for i in range(center.shape[0]):
        #     if center[i] >= 1:
        #         a_nxt[i] = 1
        #     elif center[i] < 1 and center[i] > 0:
        #         a_nxt[i] = center[i]
        #     else:
        #         a_nxt[i] = 0
        #
        # return a_nxt
