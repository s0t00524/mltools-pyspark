#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients


class BinaryLogisticRegressionLossGradFunEval(LossGradFunEval):

    def is_differential(self) -> bool:
        return True

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:
        x = data.input
        y = data.target  # -1 or 1

        dot = np.dot(param, x)
        sum_exp = 1 + math.exp(-y * dot)
        loss = math.log(sum_exp)

        grad = -y * x * math.exp(- y * dot) / sum_exp
        lg = LossAndGradients(loss, grad)
        return lg