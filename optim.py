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
    def optimize(self, init_weights, dataset,
                 f_eval: LossGradFunEval,
                 regularizer: Regularizer, lmd):
        pass


class SteepestGradientDescentOptimizer(Optimizer):
    """
    implementation of steepest gradient descent
    with differentiable regularization
    """

    def __init__(self, n_iter=1000, tol=1e-6, lr=1.0, c1=0.3, rho=0.9):
        super().__init__(n_iter, tol, lr)
        self.c1 = c1  # const. for Armijo's condition
        self.rho = rho  # updating ratio for learning rate lr

        self.optimized_weights = None
        self.optimized_results = None
        self.path = None

    def set_max_iter(self, n_iter: int):
        assert n_iter > 0
        self.n_iter = n_iter

    def set_tolerance(self, tol: float):
        assert tol >= 1e-6
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

        assert regl.is_differential()

        w = init_weights
        eta = self.lr
        path = [w]

        for i in range(self.n_iter):

            loss_grad = f_eval.get_loss_grad(w, dataset)
            loss = loss_grad.loss
            grad = loss_grad.grad

            loss_with_regl = loss + lmd * regl.get_score(w)
            grad_with_regl = grad + lmd * regl.get_grad(w)
            grad_norm_with_regl = np.linalg.norm(grad_with_regl)

            # termination condition
            if grad_norm_with_regl < self.tol:
                break

            # calculate direction based on inverse of B
            direction = - grad_with_regl

            # calculate param. in the next step
            _w_next = w + eta * direction

            # line search for eta
            _next_loss = f_eval.get_loss_grad(_w_next, dataset).loss
            _next_grad = f_eval.get_loss_grad(_w_next, dataset).grad
            _next_loss_with_regl = _next_loss + lmd * regl.get_score(_w_next)
            _next_grad_with_regl = _next_grad + lmd * regl.get_grad(_w_next)
            _next_grad_norm_with_regl = np.linalg.norm(_next_grad_with_regl)

            while _next_loss_with_regl > loss_with_regl + self.c1 * eta * np.dot(grad_with_regl, direction):
                eta = self.rho * eta

            # update parameter
            s_k = eta * direction
            w = w + s_k

            path.append(w)

            print("[iter: %04d] [loss: %.6f] [norm of gradient: %.6f] eta: %.6f"
                  % (i, loss_with_regl, grad_norm_with_regl, eta))

        # process for termination
        final_loss = f_eval.get_loss_grad(w, dataset).loss + lmd * regl.get_score(w)

        self.optimized_weights = w
        self.optimized_results = final_loss
        self.path = np.array(path)


class ProximalOptimizer(Optimizer):
    """
    implementation of steepest gradient descent with proximal optimization
    """

    def __init__(self, n_iter=1000, tol=1e-5, lr=1.0):
        super().__init__(n_iter, tol, lr)

        self.optimized_weights = None
        self.optimized_results = None
        self.path = None

    def set_max_iter(self, n_iter: int):
        assert n_iter > 0
        self.n_iter = n_iter

    def set_tolerance(self, tol: float):
        assert tol >= 1e-6
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

            w_old = w

            loss_grad = f_eval.get_loss_grad(w, dataset)
            loss = loss_grad.loss
            grad = loss_grad.grad
            grad_norm = np.linalg.norm(grad)

            direction = -grad

            # calculate temporal param. in the next step
            _w_next = w + eta * direction  # TODO

            # line search for eta
            _loss_next = f_eval.get_loss_grad(_w_next, dataset).loss
            while True:
                _upper_bound = loss + np.dot(grad, _w_next - w) + np.linalg.norm(_w_next - w, 2) ** 2 / (2 * eta)

                if _loss_next <= _upper_bound:
                    break
                else:
                    eta = self.rho * eta

            # calculate param. in the next step
            w_next = w + eta * direction

            # proximal operation
            w_next = regl.prox(w_next, eta * lmd)

            w = w_next
            path.append(w)

            # calculate diff of norm of gradient

            grad_diff = w - w_old
            norm_diff = np.linalg.norm(grad_diff)

            print("[iter: %04d] [diff of weight: %.6f] [loss: %.6f] [norm of gradient: %.6f] loss_next: %.6f, "
                  "upper_bound: %.6f, eta: %.6f "
                  % (i, norm_diff, loss, grad_norm, _loss_next, _upper_bound, eta))

            # termination condition
            if norm_diff < self.tol:
                break

        # process for termination
        final_loss = f_eval.get_loss_grad(w, dataset).loss

        self.optimized_weights = w
        self.optimized_results = final_loss
        self.path = np.array(path)



class BFGSOptimizer(Optimizer):
    """
    BFGS method is one of the quasi-Newton methods.
    It can derive B, an approximation of Hessian matrix efficiently.

    This implementation is not support proximal operation
    """

    def __init__(self, n_iter=1000, tol=1e-6, lr=1.0, c1=0.3, rho=0.9):
        super().__init__(n_iter, tol, lr)
        self.c1 = c1  # const. for Armijo's condition
        self.rho = rho  # updating ratio for learning rate lr

        self.optimized_weights = None
        self.optimized_results = None
        self.path = None

    def set_max_iter(self, n_iter: int):
        assert n_iter > 0
        self.n_iter = n_iter

    def set_tolerance(self, tol: float):
        assert tol >= 1e-6
        self.tol = tol

    def optimize(self, init_weights, dataset,
                 f_eval: LossGradFunEval,
                 regl: Regularizer, lmd):

        assert regl.is_differential()

        w = init_weights
        eta = self.lr
        path = [w]

        w_dim = w.shape[0]
        B_inv = np.eye(w_dim)  # initialize B by identity matrix

        for i in range(self.n_iter):

            loss_grad = f_eval.get_loss_grad(w, dataset)
            loss = loss_grad.loss
            grad = loss_grad.grad

            loss_with_regl = loss + lmd * regl.get_score(w)
            grad_with_regl = grad + lmd * regl.get_grad(w)
            grad_norm_with_regl = np.linalg.norm(grad_with_regl)

            # termination condition
            if grad_norm_with_regl < self.tol:
                break

            # calculate direction based on inverse of B
            direction = - np.dot(B_inv, grad_with_regl)

            # calculate param. in the next step
            _w_next = w + eta * direction

            # # proximal operation
            # _w_next = regl.prox(w_next, eta * lmd)
            #
            # delta_w = _w_next - w
            #
            # # line search for eta
            # _loss_next = f_eval.get_loss_grad(w_next, dataset).loss
            # while True:
            #     # upper_bound = loss + np.dot(grad, w_next - w) + np.linalg.norm(w_next - w, 2) ** 2 / (2 * eta)
            #     upper_bound = loss + np.dot(grad, w_next - w) + 0.5 * np.dot(w_next - w, np.dot(B_inv, w_next - w)) / eta
            #
            #     if _loss_next <= upper_bound:
            #         break
            #     else:
            #         eta = self.rho * eta

            # line search for eta
            _next_loss = f_eval.get_loss_grad(_w_next, dataset).loss
            _next_grad = f_eval.get_loss_grad(_w_next, dataset).grad
            _next_loss_with_regl = _next_loss + lmd * regl.get_score(_w_next)
            _next_grad_with_regl = _next_grad + lmd * regl.get_grad(_w_next)
            _next_grad_norm_with_regl = np.linalg.norm(_next_grad_with_regl)

            while _next_loss_with_regl > loss_with_regl + self.c1 * eta * np.dot(grad_with_regl, direction):
                eta = self.rho * eta

            # update parameter
            s_k = eta * direction
            w = w + s_k

            # update inverse of B based on y and s
            grad_next = f_eval.get_loss_grad(w, dataset).grad
            y_k = grad_next - grad

            yB_dot_k = np.dot(y_k, B_inv)
            By_dot_k = np.dot(B_inv, y_k)

            yBy_dot_k = np.dot(yB_dot_k, y_k)

            sy_dot_k = np.dot(s_k, y_k)

            ss_outer_k = np.outer(s_k, s_k)
            Bys_outer_k = np.outer(By_dot_k, s_k)
            syB_outer_k = np.outer(s_k, yB_dot_k)

            F_k = (sy_dot_k + yBy_dot_k) * ss_outer_k / (sy_dot_k ** 2)
            G_k = (Bys_outer_k + syB_outer_k) / sy_dot_k

            B_inv = B_inv + F_k - G_k

            print("[iter: %04d] [loss: %.6f] [norm of gradient: %.6f], eta: %.6f"
                  % (i, loss_with_regl, grad_norm_with_regl, eta))

        # process for termination
        final_loss = f_eval.get_loss_grad(w, dataset).loss + lmd * regl.get_score(w)

        self.optimized_weights = w
        self.optimized_results = final_loss
        self.path = np.array(path)
