# NeighbourGroups

## Table of contents

  * [Installation](#installation)
  * [Usage](#usage)
    * [1. Download Publication Data](#1.-download-publication-data)

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
The following command will download the data and save it to the directory `./analysis`.

```bash
ngroups getData --dir analysis
```


### 2. Split Training and Testing Data
The following command splits the example data into a training and testing data set.
The first argument of most `ngroups` commands is the `prefix` - this defines the directory and filename prefix of Neighbour Groups outputs
For example, below each output file is prefixed with `./analysis/example` (e.g. `./analysis/example-test.csv`).
*The prefix should be kept the same through a given analysis workflow.*

```bash
ngroups prepare analysis/example analysis/C.jejuni-UKisolates.csv --trainSize 0.8 --seed 42
```

If required, the user can specify which column indexes (zero-based) correspond to the isolate ID and which correspond to the relevant training features.
For example, if the isolate ID is the second column (index 1) and the features are in the third and forth columns (indices 2 and 3) then the command would be as follows.

```bash
ngroups prepare analysis/example analysis/C.jejuni-UKisolates.csv \
  --IDcol 1 --features 2 3 --trainSize 0.8 --seed 42
```


### 3. Build Phylogenetic Tree
Following splitting of the data the user must create **two** phylogenetic trees in newick format; this step must be performed externally.
The published methodology builds a Neighbour Joining (NJ) tree from the core MLST loci using [PubMLST](https://pubmlst.org/).
However, in principle, any phylogenetic approach can be used.
The key requirement is that the labels of the Newick trees match the corresponding isolate IDs of the full and training data set.

*Note: The example data download in step 1 already included pre-computed newick trees. If using the example data, skip to step 4.*

#### Full Tree
The first NJ tree is constructed from the full set of isolates - in the example these are saved to ``analysis/example-full.csv``.
The full tree will be used following model training to assess the prediction accuracy of the hold-out test set.
In addition the full tree can later be used to re-train a final model on the full data set, following validation.

#### Training Tree
The second NJ tree is constructed from the training subset of isolates - in the example these are saved to ``analysis/example-train.csv``.
The training tree is used to extract target Neigbour Groups and train the classifier model.


### 4. Pre-process the Trees

```bash
ngroups tree analysis/example data/C.jejuni-full.nwk data/C.jejuni-train.nwk
```


### 5. Training the Model
After completing the previous sets the mode can be train as follows
The number of Neighbour Groups to classify must be specified as positional arguments following the prefix and a seed can be set for reproducibility.
Multiple Neighbour Group clusters can be provided to train different models at different tree hierarchy levels.

```bash
ngroups train analysis/example 20 30 --seed 42
```

### 6. Testing the Model
Following training, the `ngroups test` command can be used to assess classifier performance.
For each Neighbour Group (e.g. 20 and 30 above) an adjusted Rand index will be computed and written to stdout.

```bash
ngroups test analysis/example > adjustedRandScores.csv
```

### 7. Re-train model with Full Data
Following testing, the model can be retrained using the full dataset.
To retrain the model re-run the `ngroups train` command from step 5 with an additional `--full` flag.
This will output a final trained model at the location `{prefix}-{nGroup}-final-trained.pkl`.
For example, in the following command the model will be written to `./analysis/example-20-final-trained.pkl`

In addition, a final CSV final will be written to `{prefix}-{nGroup}-final.csv` which includes the NG predictions and original tree groups for all of the input data.

```bash
ngroups train analysis/example 20 --full --seed 42
```

### 8. Use the Model
Now the classifier is trained, it can be used on other data.
The `ngroups predict` command requires a path to a CSV of data the trained model.

*Note: The column names of the CSV must match the column names used when training the model.*

```bash
ngroups predict data/C.jejuni-UKisolates.csv analysis/example-20-final-trained.pkl \
  > C.jejuni-UKisolates-classified.csv
```
