#!/usr/bin/env python
# coding: utf-8

import numpy as np

from tools.binary.hinge_loss_grad import HingeLossGradFunEval
from tools.dataset import PtData
from tools.loss_grad import LossGradFunEval, LossAndGradients
from tools.optim import ProjectionOptimizer
from tools.projector import ParamIn0to1Projector
from tools.regularizer import L2Regularizer


class DualLFormOfL2RegularizedHingeLossGradFun(LossGradFunEval):

    def __init__(self, lmd=None):

        if lmd is not None:
            self.lmd = lmd
        else:
            self.lmd = 0.01

        self.max_eig = None

    def set_lambda(self, lmd):
        self.lmd = lmd

    def is_differential(self) -> bool:
        return True

    def set_max_eig_of_K(self, K):
        max_eig = np.max(np.linalg.eigvals(K/(2*self.lmd)))
        self.max_eig = max_eig

    def get_max_eig_of_K(self):
        assert self.max_eig is not None
        return self.max_eig

    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:
        pass

    def get_loss_grad(self, param, data_list) -> LossAndGradients:

        data_num = len(data_list)
        K = np.zeros((data_num, data_num))

        for i, datai in enumerate(data_list):
            for j, dataj in enumerate(data_list):
                kxij = np.dot(datai.input, dataj.input)
                K[i][j] = datai.target * dataj.target * kxij

        # print(param)
        # self.set_max_eig_of_K(K)

        K_param = np.dot(K, param)
        loss = - (1. / (4 * self.lmd)) * np.dot(param, K_param) + param.sum()
        grad = (1 / (2 * self.lmd)) * K_param - np.ones((data_num,))

        return LossAndGradients(loss, grad)

class BinarySVMClassifier:

    def __init__(self, lmd):

        self.lmd = lmd

        self.primal_hist = None
        self.dual_hist = None

    def train(self, dataset, lr=0.1, lr_decay=True):

        data_list = dataset.collect()
        data_num = len(data_list)
        init_lag = np.zeros((data_num,))

        # print(data_list)

        # primal function
        primal_feval = HingeLossGradFunEval()
        l2_regl = L2Regularizer()

        # dual function
        dual_feval = DualLFormOfL2RegularizedHingeLossGradFun(lmd=self.lmd)

        # projection operator
        projector = ParamIn0to1Projector()

        # optimizer
        optimizer = ProjectionOptimizer(lr=lr, lr_decay=lr_decay, n_iter=20000)

        # optimize
        optimizer.optimize(init_lag, data_list, dual_feval, projector, self.lmd)

        self.final_lag = optimizer.optimized_weights

        optimized_lag_list = optimizer.path
        weight_path = [self.get_param_from_lagrange(lag, data_list)
                       for lag in optimized_lag_list]

        # primal loss history
        primal_hist = [primal_feval.get_sum_of_loss_grad(w, dataset).loss + 2 * self.lmd * l2_regl.get_score(w)
                       for w in weight_path]

        dual_hist = optimizer.train_log

        self.optimized_weights = self.get_param_from_lagrange(self.final_lag, data_list)
        self.primal_hist = primal_hist
        self.dual_hist = dual_hist

        print("finish learning")



    def get_param_from_lagrange(self, lag, data_list):

        score = 1 / (2 * self.lmd)
        lagyx_list = [l * d.input * d.target for l, d in zip(lag, data_list)]

        return score * sum(lagyx_list)

    def infer(self, input):
        pass









