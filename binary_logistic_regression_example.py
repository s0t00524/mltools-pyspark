#!/usr/bin/env python
# coding: utf-8
import math

import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkContext

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients
from tools.optim import SteepestGradientDescentOptimizer, BFGSOptimizer, ProximalOptimizer
from tools.regularizer import L1Regularizer, L2Regularizer
from tools.binary.binary_logistic_loss_grad import BinaryLogisticRegressionLossGradFunEval


if __name__ == '__main__':

    # launch spark
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    np.random.seed(1234)

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
    l2_regl = L2Regularizer()

    sgdOptimizer = SteepestGradientDescentOptimizer(lr=1.0)
    bfgsOptimizer = BFGSOptimizer(lr=1.0)
    proximalOperator = ProximalOptimizer(lr=1.0)

    sgdOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    bfgsOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    proximalOperator.optimize(init_weights, dataset, f_eval, l2_regl, lmd)

    optimized_weights1 = sgdOptimizer.optimized_weights
    optimized_weights2 = bfgsOptimizer.optimized_weights
    optimized_weights3 = proximalOperator.optimized_weights

    print("=================================================")
    print("Logisic Regression with L2 regularization")
    print("Gradient Descent     : ", optimized_weights1)
    print("BFGS (quasi-Newton)  : ", optimized_weights2)
    print("Proximal Operator    : ", optimized_weights3)

    # get training log of loss
    print("start plotting")
    sgd_log = np.abs(sgdOptimizer.train_log - sgdOptimizer.optimized_result)
    bfgs_log = np.abs(bfgsOptimizer.train_log - sgdOptimizer.optimized_result)

    figsize = (10, 10)
    save_base = "./out/compare-sgd-with-bfgs"

    plt.figure(figsize=figsize)
    plt.plot(sgd_log, marker="o", label="Steepest Gradient Descent Method")
    plt.plot(bfgs_log, marker="^", label="BFGS method (quasi-Newton method)")
    plt.yscale('log')

    plt.xlabel("# of iteration")
    plt.ylabel("$| J(w^{(t)}) - J(\hat{w}) |$")
    plt.title("Performance Comparison on Binary Logistic Regression")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()
