#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients


class MultiClassLogisticRegressionLossGradFunEval(LossGradFunEval):

    def is_differential(self) -> bool:
        return True

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:
        """
        return point wise loss and grad
        for multi class logistic regression

        C: number of classes
        param \in R^{C \times dim(x)} (np.array in shape (C x dim(x))
        """

        x = data.input
        y = data.target  # in [0, 1, 2, ..., C]

        C, x_dim = param.shape

        logits = np.dot(param, x)
        # print(logits)
        # print(logits.shape)
        log_sum_exp = math.log(np.sum(np.exp(logits)))
        loss = - logits[y] + log_sum_exp

        grad = []
        for i in range(C):
            post = math.exp(logits[i] - log_sum_exp)
            if i == y:
                grad_per_class = x * (post - 1.0)
                grad.append(grad_per_class)
            else:
                grad_per_class = x * post
                grad.append(grad_per_class)

        grad = np.array(grad)
        return LossAndGradients(loss, grad)