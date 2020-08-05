#!/usr/bin/env python
# coding: utf-8
from collections import namedtuple

PtData = namedtuple("PtData", ("input", "target"))

if __name__ == '__main__':

    data = PtData(input=20, target=1)
    print(data.input)
    print(data.target)
