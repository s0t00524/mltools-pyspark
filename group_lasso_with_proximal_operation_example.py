#!/usr/bin/env python
# coding: utf-8

import math

import cvxpy as cv
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkContext

from tqdm import tqdm

from tools.dataset import PtData
from tools.optim import ProximalOptimizer, AcceleratedProximalOptimizer
from tools.regularizer import GroupL1Regularizer
from tools.regress.least_square_loss_grad import LeastSquareLossGradFunEval

def mldivide(A, b):
  piA = np.linalg.pinv(A)
  x = np.dot(piA, b)
  return x

if __name__ == '__main__':

    # launch spark
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    np.random.seed(1234)

    # create dataset
    # dataset 6
    d_d6 = 200
    n_d6 = 180

    # we consider 5 groups where each group has 40 attributes
    groupSize = 5
    attrSize = 40
    g_d6 = np.arange(groupSize * attrSize).reshape(groupSize, -1)

    x_d6 = np.random.randn(n_d6, d_d6)
    noise_d6 = 0.5

    # we consider feature in group 1 and group 2 is activated.
    w_d6 = np.vstack([20 * np.random.randn(80, 1),
                      np.zeros((120, 1)),
                      5 * np.random.rand()])
    x_d6_tilde = np.hstack([x_d6, np.ones((n_d6, 1))])
    y_d6 = np.dot(x_d6_tilde, w_d6) + noise_d6 * np.random.randn(n_d6, 1)

    y = y_d6
    x_tilde = x_d6_tilde
    w = w_d6 # true weight
    noise = noise_d6
    n = n_d6
    d = d_d6
    g = g_d6

    lam = 1.0
    wridge = mldivide(np.dot(x_tilde.T, x_tilde) + lam * np.eye(d + 1),
                      np.dot(x_tilde.T, y))

    # create dataset
    dataset = []

    for input, target in zip(x_tilde, y):
        pt_data = PtData(input=input, target=target)
        dataset.append(pt_data)

    # dataset = np.array(dataset)
    dataset = sc.parallelize(dataset)

    init_weights = np.zeros((w.shape[0],))

    f_eval = LeastSquareLossGradFunEval()
    regul = GroupL1Regularizer(g)

    # optimizer = ProximalOptimizer(lr=0.1, lr_decay=False)
    optimizer = AcceleratedProximalOptimizer(lr=0.25, lr_decay=False)

    # optimize by the proximal operator
    optimizer.optimize(init_weights, dataset, f_eval, regul, lam)
    weight_prox = optimizer.optimized_weights

    # optimize by cvtopt
    west = cv.Variable((d + 1, 1))
    obj_fn = \
        0.5 / n * cv.quad_form(x_tilde * west - y, np.eye(x_tilde.shape[0])) + \
        lam * (cv.norm(west[g[0]], 2.0) +
               cv.norm(west[g[1]], 2.0) +
               cv.norm(west[g[2]], 2.0) +
               cv.norm(west[g[3]], 2.0) +
               cv.norm(west[g[4]], 2.0))

    objective = cv.Minimize(obj_fn)
    constraints = []

    prob = cv.Problem(objective, constraints)
    result = prob.solve(solver=cv.CVXOPT)

    #### test
    x_test = np.random.randn(n, d)
    x_test_tilde = np.hstack([x_test, np.ones((n, 1))])
    y_test = np.dot(x_test_tilde, w) + noise * np.random.randn(n, 1)
    y_pred_by_cvtopt = np.dot(x_test_tilde, west.value)
    y_pred_by_prox = np.dot(x_test_tilde, weight_prox).reshape(-1, 1)

    res = np.mean((weight_prox - west.value.reshape((201,))) ** 2)

    print("cvxpy: ", np.mean((y_pred_by_cvtopt - y_test) ** 2))
    print("prox : ", np.mean((y_pred_by_prox - y_test) ** 2))
    print("residual: ", res)

    # print(weight_prox)
    # print(west.value.reshape((201,)))

    figsize = (10, 10)
    save_base = "./out/compare-grouplasso-by-prox-and-cvxpy"

    plt.figure(figsize=figsize)
    plt.plot(west.value[0:d], 'r-o', markersize=2.5, linewidth=1.5, label="w by cvtopt")
    plt.plot(w, 'b-*', markersize=1.5, linewidth=0.5, label="true w")
    plt.plot(wridge, 'g-+', markersize=1.5, linewidth=0.5, label="weidge")
    plt.plot(weight_prox, 'y-+', markersize=2.5, linewidth=1.5, label="w by pgd")
    plt.legend()

    plt.xlabel("weight index")
    plt.ylabel("value of weight")
    plt.title("Visualization of grouped weights")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()