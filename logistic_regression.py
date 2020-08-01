#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np
from pyspark import SparkContext

from dataset import PtData
from loss_grad import LossGradFunEval, LossAndGradients
from optim import SteepestGradientDescentOptimizer
from regularizer import L2Regularizer


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


if __name__ == '__main__':

    sc = SparkContext()

    np.random.seed(122)

    x_dim = 4
    N = 300
    X = 3 * (np.random.rand(N, x_dim) - 0.5)
    y = (2 * X[:, 0] - 1 * X[:, 1] + 0.5 + 0.5 * np.random.randn(N, )) > 0
    y = 2 * y - 1

    # create dataset
    dataset = []

    for input, target in zip(X, y):
        pt_data = PtData(input=input, target=target)
        dataset.append(pt_data)

    # dataset = np.array(dataset)
    dataset = sc.parallelize(dataset)

    # print(dataset.first())

    # initial parameter
    init_weights = np.zeros((x_dim,))

    lmd = 0.01  # for ridge

    f_eval = BinaryLogisticRegressionLossGradFunEval()
    regl = L2Regularizer()
    optimizer = SteepestGradientDescentOptimizer(lr=1.0)

    optimizer.optimize(init_weights, dataset, f_eval, regl, lmd)




