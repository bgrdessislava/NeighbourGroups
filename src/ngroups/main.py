#!/usr/bin/env python3

from .utils import *

import os
import sys
import glob
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score

def splitTestTrain(
        prefix: str, data: str, IDcol: int = 0, features: list = None,
        missingVal: str = 'unknown', trainSize: float  = 0.8, seed: int = 42):
    if not (0 < trainSize < 1):
        logging.error(f'--trainSize {trainSize} not in range (0, 1).')
        return 1
    if (features is not None) and (IDcol in features):
        logging.error(f'Index columns {IDcol} cannot be a feature.')
        return 1
    data = pd.read_csv(data).astype(str)
    if features is None:
        data = data.set_index(data.columns[IDcol])
    else:
        featureNames = [data.columns[i] for i in features]
        data = data.set_index(data.columns[IDcol])[featureNames]

    data.index = data.index.rename('id')
    data = data.fillna(missingVal)

    y = list(range(len(data)))
    X_train, X_test, _, _ = (
        train_test_split(
            data, list(range(len(data))),
            train_size=trainSize, random_state=seed)
    )
    data.to_csv(f'{prefix}-full.csv')
    X_train.to_csv(f'{prefix}-train.csv')
    X_test.to_csv(f'{prefix}-test.csv')


def prepTree(prefix: str, fullTree: str, trainTree: str):
    """ Save Newick format as linkage matrix """
    info = ({
        f'{prefix}-full': fullTree,
        f'{prefix}-train': trainTree,
    })
    for name, tree in info.items():
        linkageMatrix, labels = nwk2linkage(tree)
        np.save(f'{name}-linkage.npy', linkageMatrix)
        np.save(f'{name}-labels.npy', labels)


def trainAll(prefix: str, nGroup: list, full: bool = False, seed: int = 42):
    """ Wrapper to training all nGroup models """
    mode = 'full' if full else 'train'
    # Read data
    data = pd.read_csv(f'{prefix}-{mode}.csv').astype(str).set_index('id')
    linkageMatrix, labels = readTree(prefix, mode)
    for ng in nGroup:
        if ng < 2:
            logging.error(f'Neighbour Group {ng} must be 2 or more - skipping.')
            continue
        else:
            logging.info(f'Training Neighbour Group {ng}.')
            trainNG(prefix, data.copy(), linkageMatrix, labels, ng, full, seed)


def testAll(prefix: str):
    """ Wrapper to testing all trained nGroup models """
    data = readFull(prefix)
    if data['test'].sum() == 0:
        logging.error('No valid test data detected - exciting')
        return 1
    # Process full tree and add fullNG labels
    linkageMatrix, labels = readTree(prefix, mode='full')

    print('NeighbourGroup', 'AdjRand', sep=',')
    models = glob.glob(f'{prefix}-*-trained.pkl')
    for model in models:
        nGroup = model.split('-')[-2]
        if nGroup == 'final':
            continue
        adjRand = testNG(prefix, data.copy(), linkageMatrix, labels, nGroup)
        print(nGroup, adjRand, sep=',')


def runNG(model: str, data: str, col: str = 'NG'):
    """ Generate NG classifications using trained model """
    data = pd.read_csv(data).astype(str)
    model = readPickle(model)
    # Dont overwrite an excisting column
    assert col not in data.columns
    data[col] = model.predict(data)
    data.to_csv(sys.stdout, index=False)


def downloadExample(dir: str = '.'):
    """ Download example dataset from GitHub repo """
    prefix = 'https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data'
    download(f'{prefix}/C.jejuni-UKisolates.csv', dir)
    download(f'{prefix}/C.jejuni-full.nwk', dir)
    download(f'{prefix}/C.jejuni-train.nwk', dir)


def downloadModel(path: str = '.'):
    """ Download pre-trained model from publication. """
    prefix = 'https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data'
    logging.error('Not Implemented')
    return 1
    download(f'{prefix}/C.jejuni-UKisolates.pkl', dir)
