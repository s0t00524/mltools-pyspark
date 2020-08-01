#!/usr/bin/env python
# coding: utf-8

from abc import abstractmethod, ABCMeta

import numpy as np

from loss_grad import LossGradFunEval
from regularizer import Regularizer


class Optimizer(metaclass=ABCMeta):

    def __init__(self, n_iter=1000, tol=1e-6, lr=0.001):
        assert n_iter > 0
        self.n_iter = n_iter

        assert tol >= 1e-6
        self.tol = tol

        assert lr > 0
        self.lr = lr

        self.path = None  # list of param to be being optimized

    @abstractmethod
    def set_max_iter(self, n_iter):
        pass

    @abstractmethod
    def set_tolerance(self, tol):
        pass

    @abstractmethod
    def optimize(self, init_weight, dataset,
                 f_eval: LossGradFunEval,
                 regularizer: Regularizer, lmd):
        pass


class SteepestGradientDescentOptimizer(Optimizer):

    def __init__(self, n_iter=1000, tol=1e-6, lr=1.0, c1=0.3, rho=0.9):
        super().__init__(n_iter, tol, lr)
        self.c1 = c1  # const. for Armijo's condition
        self.rho = rho  # updating ratio for learning rate lr

    def set_max_iter(self, n_iter: int):
        assert n_iter > 0
        self.n_iter = n_iter

    def set_tolerance(self, tol: float):
        assert tol > 1e-6
        self.tol = tol

    def optimize(self, init_weights, dataset, f_eval: LossGradFunEval,
                 regl: Regularizer, lmd: float):
        """
        optimize weights by steepest gradient method
        :param init_weights:
        :param dataset:
        :param f_eval:
        :param regl:
        :param lmd:
        :return:
        """

        w = init_weights
        eta = self.lr
        path = [w]

        for i in range(self.n_iter):

            loss_grad = f_eval.get_loss_grad(w, dataset)
            loss = loss_grad.loss
            grad = loss_grad.grad

            # termination condition
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                break

            direction = -grad

            # calculate temporal param. in the next step
            _w_next = w + eta * direction

            # line search for eta
            while True:
                _loss_next = f_eval.get_loss_grad(_w_next, dataset).loss
                _upper_bound = loss + np.dot(grad, _w_next - w) + np.linalg.norm(_w_next - w, 2) ** 2 / (2 * eta)

                if _loss_next <= _upper_bound:
                    break
                else:
                    eta = self.rho * eta

            # calculate param. in the next step
            w_next = w + eta * direction

            # proximal operation
            w_next = regl.prox(w_next, eta * lmd)

            # _loss_next = f_eval.get_loss_grad(w_next, dataset).loss
            # _upper_bound = loss + np.dot(grad, w_next - w) + np.linalg.norm(w_next - w, 2) ** 2 / (2 * eta)
            #
            # # Search new lr until satisfy Armijo's condition
            # while _loss_next > _upper_bound:
            #     print(eta)
            #     eta = self.rho * eta
            #     _upper_bound = loss + np.dot(grad, w_next - w) + np.linalg.norm(w_next - w, 2) ** 2 / (2 * eta)

            # _loss_next = f_eval.get_loss_grad(w_next, dataset).loss
            # while _loss_next > loss + self.c1 * eta * np.dot(grad, direction):
            #     print(eta)
            #     eta = self.rho * eta

            w = w_next
            path.append(w)

            # print the current status
            print("[iter: %04d] [loss: %.6f] [norm of gradient: %.6f] loss_next: %.6f, upper_bound: %.6f, eta: %.6f"
                  % (i, loss, grad_norm, _loss_next, _upper_bound, eta))

            # print("[iter: %04d] [loss: %.6f] [norm of gradient: %.6f] ,eta: %.6f"
            #       % (i, loss, grad_norm, eta))

        # process for termination
        final_loss = f_eval.get_loss_grad(w, dataset).loss

        self.optimized_weights = w
        self.optimized_results = final_loss
        self.path = np.array(path)
