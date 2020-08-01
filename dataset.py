#!/usr/bin/env python
# coding: utf-8
from collections import namedtuple
from typing import TypeVar


T = TypeVar('T')
V = TypeVar('V')

PtData = namedtuple("PtData", ("input", "target"))

if __name__ == '__main__':

    data = PtData(input=20, target=1)
    print(data.input)
    print(data.target)
