#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients

class HingeLossGradFunEval(LossGradFunEval):

    def is_differential(self) -> bool:
        return False

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:

        x = data.input
        y = data.target  # -1 or 1

        wx = np.dot(param, x)
        ywx = y * wx

        loss = max(0.0, 1.0 - ywx)
        grad = np.zeros((len(param),))

        if loss > 0:
            grad = -y * x

        return LossAndGradients(loss, grad)

