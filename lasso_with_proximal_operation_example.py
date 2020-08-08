#!/usr/bin/env python
# coding: utf-8

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext
from tqdm import tqdm

from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients
from tools.optim import ProximalOptimizer
from tools.regularizer import L1Regularizer


class LassoPOExampleLossGradFunEval(LossGradFunEval):

    def is_differential(self) -> bool:
        return True

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:

        A = np.array([
            [3.0, 0.5],
            [0.5, 1.0]
        ])

        mu = np.array([1.0, 2.0])

        loss = np.dot(param - mu, np.dot(A, (param - mu)))
        grad = np.dot(A + A.T, param - mu)

        return LossAndGradients(loss, grad)

if __name__ == '__main__':

    # launch spark
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    np.random.seed(1234)

    # dummy data
    dataset = [PtData(input=1.0, target=1.0)]
    dataset = sc.parallelize(dataset)

    w_dim = 2
    init_weights = np.zeros((w_dim,))

    lmds = [2., 4., 6.]  # for lasso

    A = np.array([
        [3.0, 0.5],
        [0.5, 1.0]
    ])

    mu = np.array([1.0, 2.0])

    eigvals_of_A = np.linalg.eigvals(2 * A)
    lr_inv = eigvals_of_A.max()
    lr = 1. / lr_inv

    # print(lr, type(lr))

    optimized = []
    optimized_paths = []
    weight_logs = []

    f_eval = LassoPOExampleLossGradFunEval()
    l1_regl = L1Regularizer()

    for lmd in lmds:

        # optimize by implemented PO
        proximalOperator = ProximalOptimizer(lr=lr, lr_decay=False)
        proximalOperator.optimize(init_weights, dataset, f_eval, l1_regl, lmd)

        optimized_weights = proximalOperator.optimized_weights

        print("=================================================")
        print("lambda for L1 regul  : ", lmd)
        print("Proximal Operator    : ", optimized_weights)

        optimized.append(optimized_weights)
        optimized_paths.append(proximalOperator.path)
        norm_diff = [np.linalg.norm(w_t - optimized_weights) for w_t in proximalOperator.path]
        weight_logs.append(norm_diff)

        # optimize by cvxpy
        w_cvx = cp.Variable(2)

        # obj_fun = (mu.T - w_cvx.T) @ A @ (mu - w_cvx) + lmd * cp.norm1(w_cvx)
        obj_fun = cp.quad_form(w_cvx - mu, A) + lmd * cp.norm1(w_cvx)
        obj = cp.Minimize(obj_fun)

        prob = cp.Problem(obj)
        prob.solve(solver=cp.CVXOPT)

        print("Optimized by cvxpy   : ", w_cvx.value)

    # plot regularization results
    num_points = 22
    w1 = np.linspace(-2, 2, num_points)
    w2 = np.linspace(-1, 3, num_points)
    w1_mesh, w2_mesh = np.meshgrid(w1, w2)

    z = np.zeros((num_points, num_points))
    for i in tqdm(range(num_points)):
        for j in range(num_points):
            _w1 = w1_mesh[i][j]
            _w2 = w2_mesh[i][j]
            _w_comb = np.array([_w1, _w2])
            z[i][j] = f_eval.get_loss_grad(param=_w_comb, dataset=dataset).loss

    figsize = (5, 5)
    save_base = "./out/Lasso-PO-optimal-parameters"

    plt.figure(figsize=figsize)
    plt.contour(w1, w2, z, levels=np.logspace(-0.3, 1.2, 10), alpha=0.3)
    plt.scatter(optimized[0][0], optimized[0][1], s=100, marker="*", label="$\lambda = 2$")
    plt.scatter(optimized[1][0], optimized[1][1], s=100, marker="*", label="$\lambda = 4$")
    plt.scatter(optimized[2][0], optimized[2][1], s=100, marker="*", label="$\lambda = 6$")

    plt.grid()
    plt.xlabel("$\hat{w_1}$")
    plt.ylabel("$\hat{w_2}$")
    plt.title("Optimized $w$ with Lasso ($\lambda$)")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()

    # plot optimization path
    figsize = (10, 10)
    save_base = "./out/Lasso-PO-optimization-path"

    plt.figure(figsize=figsize)
    plt.contour(w1, w2, z, levels=np.logspace(-0.3, 1.2, 10), alpha=0.3)
    plt.plot(optimized_paths[0][:,0], optimized_paths[0][:,1], marker="o", label="$\lambda = 2$")
    plt.plot(optimized_paths[1][:,0], optimized_paths[1][:,1], marker="o", label="$\lambda = 4$")
    plt.plot(optimized_paths[2][:,0], optimized_paths[2][:,1], marker="o", label="$\lambda = 6$")

    plt.grid()
    plt.xlabel("$\hat{w_1}$")
    plt.ylabel("$\hat{w_2}$")
    plt.title("Optimization path of $w$ with Lasso ($\lambda$)")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()

    # visualize norm of w - w
    figsize = (10, 10)
    save_base = "./out/transiton-of-norm"

    plt.figure(figsize=figsize)
    plt.plot(weight_logs[0], marker="*", label="$\lambda = 2$")
    plt.plot(weight_logs[1], marker="*", label="$\lambda = 4$")
    plt.plot(weight_logs[2], marker="*", label="$\lambda = 6$")
    plt.yscale('log')

    plt.xlabel("# of iteration")
    plt.ylabel("$||w^{(t)} - \hat{w})||$")
    plt.title("$||w^{(t)} - \hat{w})||$ w.r.t. # of iterations")
    plt.legend(loc="upper right")
    plt.savefig(save_base + ".png")
    plt.savefig(save_base + ".eps")
    plt.savefig(save_base + ".pdf")
    plt.close()

    # visualize norm of w_1
    figsize = (5, 5)
    save_base = "./out/w-and-lambda-on-lasso"

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot([2, 4, 6], [optimized[0][0], optimized[1][0], optimized[2][0]], "b-*", label="$w_1$")

    ax1.set_xlabel("$\lambda$")
    ax1.set_ylabel("$w_1$")
    ax1.set_title("$w_1$ w.r.t. $\lambda$")
    ax1.legend(loc="upper right")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.plot([2, 4, 6], [optimized[0][1], optimized[1][1], optimized[2][1]], "g-*", label="$w_2$")

    ax2.set_xlabel("$\lambda$")
    ax2.set_ylabel("$w_2$")
    ax2.set_title("$w_2$ w.r.t. $\lambda$")
    ax2.legend(loc="upper right")
    ax2.grid()

    fig.savefig(save_base + ".png")
    fig.savefig(save_base + ".eps")
    fig.savefig(save_base + ".pdf")
    fig.close()




