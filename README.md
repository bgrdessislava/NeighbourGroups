# NeighbourGroups

## Table of contents

  * [Installation](#installation)
  * [Usage](#usage)
    * [1. Download Publication Data](#1-download-publication-data)
    * [2. Split Training and Testing Data](#2-split-training-and-testing-data)
    * [3. Build Phylogenetic Trees](#3-build-phylogenetic-tree)
    * [4. Pre-process the Trees](#4-pre-process-the-trees)
    * [5. Training the Model](#5-training-the-model)
    * [6. Testing the Model](#6-testing-the-model)
    * [7. Re-train the Model with Full Data](#7-re-train-the-model-with-full-data)
    * [8. Using the Model](#8-using-the-model)
  * [Downstream Analysis](#downstream-analysis)
    * [Investigating Neighbour Group Relationships](#investigating-neighbour-group-relationships)


## Installation

```bash
pip install git+https://github.com/bgrdessislava/NeighbourGroups.git
```

## Usage
Neighbour Groups can be run from the command line and additional help is provided via ```ngroups --help```.
Each sub-command also has help documentation (e.g. `ngroups train --help`).
The following commands can be used to reproduce the findings of the publication.

### 1. Download Publication Data
The publication data can be downloaded as below or obtained directly from the [GitHub repository](https://github.com/bgrdessislava/NeighbourGroups/tree/main/data).
The following command will download the data and save it to the directory `./data`.

```bash
ngroups getData --dir data
```

### 2. Split Training and Testing Data
The following command splits the example data into a training and testing data set.
The first argument of most `ngroups` commands is the `prefix` - this defines the directory and filename prefix of Neighbour Groups outputs
For example, below each output file is prefixed with `./example` (e.g. `./example-test.csv`).
The prefix should be kept the same through a given analysis workflow.

```bash
ngroups prepare example data/C.jejuni-UKisolates.csv --trainSize 0.8 --seed 42
```

If required, the user can specify which columns correspond to the isolate ID and which correspond to the relevant training features.
For example, if the isolate ID is labelled `id` and we want to only use `aspA` and `glnA` as features then the command would be as follows.
If no values are specified, the isolated ID is assumed to be the first column and all subsequent columns are treated as features.

```bash
ngroups prepare example data/C.jejuni-UKisolates.csv \
  --IDcol id --features aspA glnA --trainSize 0.8 --seed 42
```

### 3. Build Phylogenetic Trees
Following splitting of the data the user must create **two** phylogenetic trees in newick format; this step must be performed externally.
The published methodology builds a Minnimum Spamming tree (MST) from the core MLST loci using [PubMLST](https://pubmlst.org/).
However, in principle, any phylogenetic approach can be used.
The key requirement is that the labels of the Newick trees match the corresponding isolate IDs of the full and training data set.

*Note: The example data download in step 1 already included pre-computed newick trees. If using the example data, skip to step 4.*

#### Full Tree
The first tree is constructed from the full set of isolates - in the example these are saved to ``output/example-full.csv``.
The full tree will be used following model training to assess the prediction accuracy of the hold-out test set.
In addition the full tree can later be used to re-train a final model on the full data set, following validation.

#### Training Tree
The second tree is constructed from the training subset of isolates - in the example these are saved to ``output/example-train.csv``.
The training tree is used to extract target Neigbour Groups and train the classifier model.


### 4. Pre-process the Trees

```bash
ngroups tree example data/C.jejuni-full.nwk data/C.jejuni-train.nwk
```

### 5. Training the Model
After completing the previous steps the mode can be trained as follows.
The number of Neighbour Groups to classify must be specified as positional arguments following the prefix and a seed can be set for reproducibility.
Multiple Neighbour Group clusters can be provided to train different models at different tree hierarchy levels.

```bash
ngroups train example 20 30 --seed 42
```

### 6. Testing the Model
Following training, the `ngroups test` command can be used to assess classifier performance.
For each Neighbour Group (e.g. 20 and 30 above) an adjusted Rand index will be computed and written to stdout.

```bash
ngroups test example > adjustedRandScores.csv
```

### 7. Re-train the Model with Full Data
Following testing, the model can be retrained using the full dataset.
To retrain the model re-run the `ngroups train` command from step 5 with an additional `--full` flag.
This will output a final trained model at the location `{prefix}-{nGroup}-final-trained.pkl`.
For example, in the following command the model will be written to `./example-20-final-trained.pkl`

In addition, a final CSV final will be written to `{prefix}-{nGroup}-final.csv` which includes the NG predictions and original tree groups for all of the input data.

```bash
ngroups train example 20 --full --seed 42
```

### 8. Using the Model
Now the classifier is trained, it can be used on other data.
The `ngroups predict` command requires a path to the data (CSV format) and the trained model.

*Note: The header names of the CSV must include the features names used when training the model.*

```bash
ngroups predict data/C.jejuni-UKisolates.csv example-20-final-trained.pkl \
  > C.jejuni-UKisolates-classified.csv
```

## Downstream Analysis

### Investigating Neighbour Group Relationships
After training the final model (step 7) the `ngroups stats` command can be used to assess relationships between Neighbour Group clusters.
This tool computes the mean distance between isolates of each pair of Neighbour Groups.
A heatmap (SVG) and CSV file are written to `{prefix}-{nGroup}-meanNGdist.*`, as below.

```bash
ngroups stats example
```
![ngdist](./README_files/example-20-meanNGdist.svg)
