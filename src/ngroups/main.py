#!/usr/bin/env python3

from .utils import *
import sys
import logging
import pandas as pd


def splitTestTrain(data: str, prefix: str, trainSize: float):
    if not (0 < trainSize < 1):
        logging.error(f'--trainSize {trainSize} not in range (0, 1).')
        return 1


def trainNG(newick: str, prefix: str, nGroup: int):
    if nGroup < 2:
        logging.error(f'--nGroup {nGroup} must be 2 or more.')
        return 1


def testNG(newick: str, prefix: str):
    pass


def runNG(model: str, data: str):
    pass


def downloadExample(path: str):
    """ Download example dataset from GitHub repo """
    print('Downloading...')


def downloadModel(path: str):
    """ Download pre-trained model from publication. """
    print('Downloading...')
