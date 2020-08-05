#!/usr/bin/env python
# coding: utf-8
from abc import abstractmethod, ABCMeta
from typing import Union

from pyspark import RDD

from tools.dataset import PtData


class LossAndGradients(object):
    """
    class to hold loss and grad information
    """

    def __init__(self, loss, grad):
        self.loss = loss
        self.grad = grad


class LossGradFunEval(metaclass=ABCMeta):
    """
    abstract class for loss and its grad
    You should extends this class to implement
    your own loss function and its gradient.
    """

    @abstractmethod
    def is_differential(self) -> bool:
        pass

    @abstractmethod
    def _ptwise_loss_and_grad(self, param, data: PtData) -> LossAndGradients:
        pass

    def get_loss_grad(self, param, dataset#: Union[RDD, PtData]
                      ) -> LossAndGradients:

        # n_dataset = len(dataset)
        n_dataset = dataset.count()

        def loss_grad_f(data: PtData):
            return self._ptwise_loss_and_grad(param, data)

        loss_grad_per_pt = dataset.map(loss_grad_f)

        lg_total = loss_grad_per_pt.reduce(
            lambda lg1, lg2: LossAndGradients(lg1.loss + lg2.loss, lg1.grad + lg2.grad)
        )  # TODO

        # print(lg_total.loss)

        lg = LossAndGradients(loss=lg_total.loss / n_dataset , grad=lg_total.grad / n_dataset)
        return lg

    def get_sum_of_loss_grad(self, param, dataset#: Union[RDD, PtData]
                      ) -> LossAndGradients:

        # n_dataset = len(dataset)
        n_dataset = dataset.count()

        def loss_grad_f(data: PtData):
            return self._ptwise_loss_and_grad(param, data)

        loss_grad_per_pt = dataset.map(loss_grad_f)

        lg_total = loss_grad_per_pt.reduce(
            lambda lg1, lg2: LossAndGradients(lg1.loss + lg2.loss, lg1.grad + lg2.grad)
        )  # TODO

        # print(lg_total.loss)

        lg = LossAndGradients(loss=lg_total.loss, grad=lg_total.grad)
        return lg
