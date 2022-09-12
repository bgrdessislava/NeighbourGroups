# NeighbourGroups

## Table of contents

  * [Installation](#installation)
  * [Configuration](#configuration)
  * [Usage](#usage)
  * [Example output](#example-output)
  * [References](#references)

## Installation

```bash
pip install git+https://github.com/bgrdessislava/NeighbourGroups.git
```

## Usage
Neighbour Groups can be run from the command line and additional help is provided via ```ngroups --help```.

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

### 4. Training the Model
To train the model, provide the Newick format **training tree** and the same prefix (e.g. `analysis/example`) used in steps 1 and 2.
The number of Neighbour Groups to classify can be specified (default = 20) and a seed can be set for reproducibility.

```bash
ngroups train analysis/example analysis/C.jejuni-train.nwk --nGroup 20 --seed 42
```
