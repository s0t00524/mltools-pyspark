#!/usr/bin/env python
# coding: utf-8
import math
from abc import abstractmethod, ABCMeta

import numpy as np


class Regularizer(metaclass=ABCMeta):
    """
    Abstract class for regularization.
    All regularization process should be conducted
    via proximal operation.
    """

    @abstractmethod
    def get_score(self, w) -> float:
        pass

    @abstractmethod
    def get_grad(self, w):
        pass

    @abstractmethod
    def is_differential(self) -> bool:
        pass

    @abstractmethod
    def prox(self, y, lmd):
        """
        update param. w' by the proximal operation
                                    1
        w' = argmin_w lmd * g(w) + --- || w - y ||^2
                                    2
        :param y:
        :param lmd: lambda
        :return:
        """
        pass


class L2Regularizer(Regularizer):

    def get_score(self, w) -> float:
        return np.dot(w, w) / 2.

    def get_grad(self, w):
        return w

    def is_differential(self) -> bool:
        return True

    def prox(self, y, lmd):
        return y / (1.0 + lmd)


class L1Regularizer(Regularizer):
    def get_score(self, w) -> float:
        return np.abs(w).sum()

    def get_grad(self, w):
        raise ValueError("Unsupported operation. L1 regularization term is not differentiable.")

    def is_differential(self) -> bool:
        return False

    def prox(self, y, lmd):

        dim = y.shape[0]
        param = y.copy()

        for i in range(dim):
            v = y[i]
            if np.abs(v) < lmd:
                param[i] = 0.0
            else:
                param[i] = v - np.sign(v) * lmd

        return param
