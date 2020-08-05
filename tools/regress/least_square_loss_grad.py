#!/usr/bin/env python
# coding: utf-8

import numpy as np

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients


class LeastSquareLossGradFunEval(LossGradFunEval):

    def is_differential(self) -> bool:
        return True

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:
        x = data.input
        y = data.target

        y_pred = np.dot(x, param)
        res = y_pred - y

        loss = 0.5 * res ** 2
        grad = res * x
        return LossAndGradients(loss, grad)

