#!/usr/bin/env python
# coding: utf-8

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


