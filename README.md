# Multiplicative Weights

## Setup

Requirements: [Anaconda 3](https://www.anaconda.com/distribution/) in the path.

Make sure that `conda` is available in your path. Then run `conda activate`.

**To be done when you clone or move this repo**:
```
rm -rf .condaenv
conda env remove --name mw-env || true
(cat environment.yaml && echo "prefix: $CWD/.condaenv") > .condaenv.yaml
conda env create -f .condaenv.yaml
```

**Should be done once per session:**
```
conda activate mw-env
```

**To save new dependencies**:
```
conda env export --no-builds | grep -v "prefix: " > environment.yaml
```

**To update out-of-date local environment with new dependencies in the `environment.yaml` file**:
```
conda env update -f environment.yaml --prune
```

## Usage

All modules are expected to be run from the root directory of the repository

```
python -m linreg.main.gendata --help
python -m linreg.main.train --help
python -m linreg.main.eval --help

./scripts/run.sh  1000 100 100 1000 1000 # n p snr iters_sgd iters_full
```
