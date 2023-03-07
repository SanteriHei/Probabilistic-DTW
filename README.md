# Probabilistic-DTW
This repository contains a Python implementation of Probabilistic dynamic time
warping algorithm. The code is currently WIP, and not everything is implemented
yet. 

The original implementation of the algorithm can be found at: https://github.com/SPEECHCOG/PDTW

If you use this code, please cite the original paper: 

Räsänen, O. & Cruz Blandon, M. A. (2020). Unsupervised Discovery of Recurring Speech Patterns using Probabilistic Adaptive Metrics. Proc. Interspeech-2020, Shanghai, China, pp. 4871–4875.

Available for download: https://arxiv.org/abs/2008.00731


## Usage
To run the algorithm, first clone the repository. Then install the required
packages by running
```shell
pip install -r requirements.txt
```

The algorithm contains 3 main-steps, each of which can be run separately.
1. Feature extraction
2. Low-resolution search
3. High-resolution candidate alignment

### Feature extraction
During the feature-extraction the following procedures are done:

1. The signals are downsampled to sampling rate of 8000
2. The downsampled signals are segmented into short windows (0.025*sampling-rate by default)
3. MFCC's are calculated. By default only 13 first coefficients are used.
4. Delta and deltas of deltas are calculated from MFCC's and they are stacked on top of the MFCC's.

Running the feature extraction can be done with the following line:
```shell
python cli.py extract-features path/to/dataset path/to/output
```

### Low-resolution search

The low-resolution search can be executed using the following:
```shell
python cli.py low-res-candidate-search path/to/features path/to/output
```


### High-resolution search
The high-resolution alignment search can be executed with: TBA


Running the whole pipeline is possible, but not recommended due to high
computational demands:


## Low res search progress 
- [x] feature extraction
- [x] Voice activity detection
- [x] random distance calculations
- [x] finding nearest segments with probabilistic distances

## High res search progress
- [ ] Random distance calculations
- [ ] Calculate the real alignment paths
