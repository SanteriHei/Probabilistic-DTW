# Probabilistic-DTW
This repository contains an implementation of Probabilistic dynamic time
warping in Python. The code is currently WIP, and not everything is implemented
yet. 

The original implementation of the algorithm can be found at: 




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

Running the feature extraction can be done with the following line:
```shell
python cli.py extract-features path/to/dataset path/to/output
```
The low-resolution search can be executed using the following:
```shell
python cli.py low-res-candidate-search path/to/features path/to/output
```

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
