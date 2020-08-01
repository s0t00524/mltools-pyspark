#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np
from pyspark import SparkContext

from dataset import PtData
from loss_grad import LossGradFunEval, LossAndGradients
from optim import SteepestGradientDescentOptimizer, BFGSOptimizer, ProximalOptimizer
from regularizer import L1Regularizer, L2Regularizer


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

    np.random.seed(0)

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
    l1_regl = L1Regularizer()
    l2_regl = L2Regularizer()

    sgdOptimizer = SteepestGradientDescentOptimizer(lr=1.0)
    bfgsOptimizer = BFGSOptimizer(lr=1.0)
    proximalOptimizer_l1 = ProximalOptimizer(lr=1.0)
    proximalOptimizer_l2 = ProximalOptimizer(lr=1.0)

    sgdOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    proximalOptimizer_l2.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    bfgsOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)

    proximalOptimizer_l1.optimize(init_weights, dataset, f_eval, l1_regl, lmd)

    optimized_weights1 = sgdOptimizer.optimized_weights
    optimized_weights2 = proximalOptimizer_l2.optimized_weights
    optimized_weights3 = bfgsOptimizer.optimized_weights

    print("=================================================")
    print("Logisic Regression with L2 regularization")
    print("Gradient Descent     : ", optimized_weights1)
    print("Proximal Optimization: ", optimized_weights1)
    print("BFGS (quasi-Newton)  : ", optimized_weights3)

    optimized_weights4 = proximalOptimizer_l1.optimized_weights

    print("=================================================")
    print("Logisic Regression with L1 regularization")
    print("Proximal Optimization: ", optimized_weights4)




