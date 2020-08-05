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
        return np.dot(w.T, w).sum() / 2.

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

        y_shape = y.shape
        if len(y_shape) == 1:
            y_num = 1
            y_dim = y_shape[0]
        else:
            y_num = y_shape[0]
            y_dim = y_shape[1]

        y = y.reshape(y_num, y_dim)
        param = y.copy()

        for i in range(y_num):
            for j in range(y_dim):
                v = y[i][j]
                if np.abs(v) < lmd:
                    param[i][j] = 0.0
                else:
                    param[i][j] = v - np.sign(v) * lmd

        if y_num == 1:
            param = param.reshape(y_dim)

        return param


class GroupL1Regularizer(Regularizer):

    def __init__(self, groups):
        """
        for i in num of groups, each group[i] provides the array
        of parameter ids that belongs to group i
        """

        self.groups = groups # array of ids
        self.num_groups = len(groups)


    def get_score(self, w) -> float:
        """
        Group L1 regulaization term is denoted as follows.
        Ω(w) = Σ_{g ∈ G}||w_g||_2

        return sum of norm of each group
        """
        # assert self.num_element == w.size

        norm_of_each_group = []
        for i in range(self.num_groups):
            param_ids = self.groups[i]
            sq_l2_sum = sum([w[j]*w[j] for j in param_ids])
            sr = math.sqrt(sq_l2_sum)
            norm_of_each_group.append(sr)

        return sum(norm_of_each_group)

    def get_grad(self, w):
        raise ValueError("Unsupported operation.")

    def is_differential(self) -> bool:
        return False

    def prox(self, y, lmd):

        norm_of_each_group = []
        for i in range(self.num_groups):
            param_ids = self.groups[i]
            sq_l2_sum = sum([y[j] * y[j] for j in param_ids])
            sr = math.sqrt(sq_l2_sum)
            norm_of_each_group.append(sr)

        y_ret = y.copy()

        for i in range(self.num_groups):
            # if norm_of_each_group[i] <= lmd:
            if norm_of_each_group[i] <= lmd * math.sqrt(len(self.groups[i])):
                for j in self.groups[i]:
                    y_ret[j] = 0.0
            else:
                for j in self.groups[i]:
                    y_ret[j] = y[j] * ((norm_of_each_group[i] - lmd) / norm_of_each_group[i])

        return y_ret






