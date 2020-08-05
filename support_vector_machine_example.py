#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext

from tools.binary.support_vector_machine import BinarySVMClassifier
from tools.dataset import PtData

if __name__ == '__main__':

    # launch spark
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    np.random.seed(1234)

    # x_dim = 4
    # N = 300
    # X = 3 * (np.random.rand(N, x_dim) - 0.5)
    # y = (2 * X[:, 0] - 1 * X[:, 1] + 0.5 + 0.5 * np.random.randn(N, )) > 0
    # y = 2 * y - 1

    n = 40
    omega = np.random.randn()
    noise = 0.8 * np.random.randn(n)

    X = np.random.randn(n, 2)
    y = (2 * (omega * X[:, 0] + X[:, 1] + noise > 0) - 1).reshape(-1, 1)

    # create dataset
    dataset = []

    for input, target in zip(X, y):
        pt_data = PtData(input=input, target=target)
        dataset.append(pt_data)

    # dataset = np.array(dataset)
    dataset = sc.parallelize(dataset)

    # print(dataset.first())

    # initial parameter
    init_weights = np.random.randn(X.shape[1])

    lmd = 1.  # for ridge

    svm = BinarySVMClassifier(lmd=lmd)

    svm.train(dataset, lr=.05, lr_decay=False)

    primal_hist = svm.primal_hist
    dual_hist = svm.dual_hist

    figsize = (10, 10)
    save_base = "./out/svm-primal-dual-comparison"

    plt.figure(figsize=figsize)
    plt.plot(primal_hist, marker="o", label="Hinge func")
    plt.plot(dual_hist, marker="^", label="Dual Lagrange func")
    # plt.yscale('log')

    plt.xlabel("# of iteration")
    plt.ylabel("Loss")
    plt.title("Loss Transition w.r.t. # of iteration")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()


    # f_eval = HingeLossGradFunEval()
    # l2_regl = L2Regularizer()

    # # sgdOptimizer = SteepestGradientDescentOptimizer(lr=1.0)
    # bfgsOptimizer = BFGSOptimizer(lr=1.0)
    # proximalOperator = ProximalOptimizer(lr=1.0)
    #
    # # sgdOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    # bfgsOptimizer.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    # proximalOperator.optimize(init_weights, dataset, f_eval, l2_regl, lmd)
    #
    # # optimized_weights1 = sgdOptimizer.optimized_weights
    # optimized_weights2 = bfgsOptimizer.optimized_weights
    # # optimized_weights3 = proximalOperator.optimized_weights
    #
    # print("=================================================")
    # print("Support Vector with Normal Hinge Loss")
    # # print("Gradient Descent     : ", optimized_weights1)
    # print("BFGS (quasi-Newton)  : ", optimized_weights2)
    # # print("Proximal Operator    : ", optimized_weights3)

    # get training log of loss
    # print("start plotting")
    # sgd_log = np.abs(sgdOptimizer.train_log - sgdOptimizer.optimized_result)
    # bfgs_log = np.abs(bfgsOptimizer.train_log - sgdOptimizer.optimized_result)
    #
    # figsize = (10, 10)
    # save_base = "./out/compare-sgd-with-bfgs"
    #
    # plt.figure(figsize=figsize)
    # plt.plot(sgd_log, marker="o", label="Steepest Gradient Descent Method")
    # plt.plot(bfgs_log, marker="^", label="BFGS method (quasi-Newton method)")
    # plt.yscale('log')
    #
    # plt.xlabel("# of iteration")
    # plt.ylabel("$| J(w^{(t)}) - J(\hat{w}) |$")
    # plt.title("Performance Comparison on Binary Logistic Regression")
    # plt.legend(loc="upper right")
    # plt.savefig(save_base + ".png")
    # plt.savefig(save_base + ".eps")
    # plt.savefig(save_base + ".pdf")
    # plt.close()