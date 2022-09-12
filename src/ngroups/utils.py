#!/usr/bin/env python3

import os
import pickle
import logging
import requests
import pandas as pd
from ete3 import ClusterTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def download(url: str, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = url.split('/')[-1]
    file_path = os.path.join(dir, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else: # HTTP status code 4XX/5XX
        logging.error(f'Download failed: status code {r.status_code}\n{r.text}')


def readNewick(nwk):
    with open(nwk) as fh:
        return fh.readline()


def nwk2linkage(newick: str):
    """ Convert newick tree into scipy linkage matrix """
    tree = ClusterTree(newick)
    cophenetic_matrix, newick_labels = tree.cophenetic_matrix()
    cophenetic_matrix = pd.DataFrame(
        cophenetic_matrix, columns=newick_labels, index=newick_labels)
    # reduce square distance matrix to condensed distance matrices
    pairwise_distances = pdist(cophenetic_matrix)
    # return linkage matrix and labels
    return linkage(pairwise_distances), list(cophenetic_matrix.columns)


def processNewick(newick: str, nGroup: int, name: str):
    nwk = readNewick(newick)
    linkageMatrix, labels = nwk2linkage(newick=nwk)
    clusters = fcluster(linkageMatrix, t=nGroup, criterion='maxclust')
    labelsID = pd.DataFrame(clusters, labels, columns=[name])
    return labelsID


def writeModel(prefix: str, model):
    with open(f'{prefix}-trained.pkl', 'wb') as fh:
        pickle.dump(model, fh)

def readModel(prefix: str):
    with open(f'{prefix}-trained.pkl', 'rb') as fh:
        return pickle.load(fh)
