#!/usr/bin/env python3

from .utils import *

import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
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
    data = pd.read_csv(data)
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



def trainNG(
        prefix: str, newick: str, nGroup: int = 20,
        full: bool = False, seed: int = 42):
    if nGroup < 2:
        logging.error(f'--nGroup {nGroup} must be 2 or more.')
        return 1
    # Read data
    data_path = f'{prefix}-full.csv' if full else f'{prefix}-train.csv'
    data = pd.read_csv(data_path).set_index('id')
    featureCols = data.columns

    labelsID = processNewick(newick, nGroup, name='trainNG')
    data = pd.merge(data, labelsID, left_index=True, right_index=True)

    y = data.pop('trainNG')
    X = data[featureCols]

    transformers = ([('categories', FunctionTransformer(), featureCols)])
    featureTransformer = ColumnTransformer(
        transformers=transformers, remainder='drop')
    model = Pipeline(steps=[
        ('columnTransform', featureTransformer),
        ('estimator',       CatBoostClassifier(
            verbose=0, random_seed=seed, cat_features=list(range(len(X))))),
    ])
    model = model.fit(X, y)
    writeModel(prefix, model)


def testNG(prefix: str, newick: str):
    # Read test data to extract isolate IDs
    testIsolates = pd.read_csv(f'{prefix}-test.csv')['id']
    # Read full data and set a test column
    data = pd.read_csv(f'{prefix}-full.csv')
    data['test'] = data['id'].apply(x in testIsolates)
    data = data.set_index('id')
    # Process full tree and add fullNG labels
    labelsID = processNewick(newick, nGroup, name='NG-truth')
    data = pd.merge(
        data, labelsID, left_index=True, right_index=True, how='outer')
    data.to_csv(f'{prefix}-final.csv')

    model = readModel(prefix)
    data['NG'] = model.predict(data)

    testSubset = data.loc[data['test']]
    adjRand = adjusted_rand_score(testSubset['NG'], testSubset['NG-truth'])
    print(adjRand)


def runNG(model: str, data: str):
    pass


def downloadExample(dir: str = '.'):
    """ Download example dataset from GitHub repo """
    prefix = 'https://raw.githubusercontent.com/bgrdessislava/NeighbourGroups/main/data'
    download(f'{prefix}/C.jejuni-UKisolates.csv', dir)
    download(f'{prefix}/C.jejuni-full.nwk', dir)
    download(f'{prefix}/C.jejuni-train.nwk', dir)


def downloadModel(path: str = '.'):
    """ Download pre-trained model from publication. """
    print('Downloading...')
