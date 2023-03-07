from utils import (combined_shape, pdist_cos, pdist_euc, mode, norm_cdf)

import math
import pathlib
import datetime
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin
import numba as nb
# from numba_stats import norm

from typing import List, Tuple
import numpy.typing as npt


# Format for the timestamps
_TS_FORMAT: str = "%H-%M-%ST%d-%m-%Y"

# Mark indexes of neighbouring frames with max value of 64 bit unsigned
# integers  (which is in index that should not appear)
# (so that the dtype can be int)
_IS_NEIGHBOUR: int = np.iinfo(np.uint64).max


@dataclass
class PDTWConfig:
    '''
    Defines the configuration used to run the PDTW.

    Attributes
    ----------
    alpha: float
        Defines the significance criterion for selecting the interesting
        segments. Can be seen as the maximum probability that a given distance
        is not smaller than the usual distances in the corpus.
    n_nearest: int
        The amount of nearest segments to hold in the memory.
    seq_shift: int
        The size of the shift between segments (in samples).
        (Corresponds to 'S' in the paper)
    n_frames: int
        The number of frames in the downsampled segments.
        (Corresponds to 'M' in the paper)
    seq_len: int
        The lenght of segments considered during the low-resolution matching.
        (Corresponds to 'L' in the paper)
    n_expansion: int
        The number of frames to expand the segments (i.e. +- n_expansion)
        During the high-resolution search. (Corresponds to 'E' in the paper)
    max_ram: int
        Defines the maximum amount of ram used for the distance calculations.
        Should be expressed in bytes.
    n_workers: int
        The amount of ram used to run the parallized regions
    verbose: bool
        If set to True, more information will be shown during the process.
    '''
    alpha: float
    n_nearest: int
    seq_shift: int
    n_frames: int
    seq_len: int
    n_expansion: int
    max_ram: int
    n_workers: int
    verbose: bool


def pdtw(features: List[npt.NDArray], config: PDTWConfig):
    '''
    Runs the whole probabilistic distance timewarping algorithm
    (low and high resolution search) for the given data-set

    Parameters
    ----------
    features: List[npt.NDArray]
        The list of features from the raw audio clips. Should have shape
        (n_features, frame-lenght)
    config: PDTWConfig
        The configuration for the run.
    '''
    ds_feats = _preprocess(
            features, config.seq_shift, config.seq_len, config.n_segments
    )

    # Extract the lenght of original samples
    dist_probs, candidates = _low_res_candidate_search_impl(
            ds_feats, features[0].shape[1]
    )

    if config.verbose:
        print(f"Found {candidates.shape[0]} candidates")

    if config.store_location is not None:
        dirpath = pathlib.Path(config.store_location)
        # Make sure that the director exists
        dirpath.mkdir(exist_ok=True, parents=True)

        ts = datetime.datetime.now().strftime(_TS_FORMAT)
        fp = dirpath / f"candidates_{ts}.npz"
        np.savez(fp, dist_probs=dist_probs, candidates=candidates)
        if config.verbose:
            print(f"Stored distance probabilities to {str(fp)!r}")

    _high_res_alignment_search_impl()


def low_res_search(
        features: List[npt.NDArray], config: PDTWConfig
        ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    '''
    Runs the low resolution candidate search for the given feature vectors.

    Parameters
    ----------
    features: List[npt.NDArray]
        The list of features representing the dataset.
    config: PDTWConfig
        The configuration for the run

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        Returns the downsampled features, the corresponding indices, distance
        probabilities, possible candidates and the indices to silent segments.
    '''
    ds_feats, ds_indices = _preprocess(
            features, config.seq_shift, config.seq_len, config.n_frames
    )

    # NOTE: Check that this is not set multiple times during the run
    nb.set_num_threads(config.n_workers)
    dist_probs, candidates, silent_indices = _low_res_candidate_search_impl(
            ds_feats, features[0].shape[0], config
    )

    return ds_feats, ds_indices, dist_probs, candidates, silent_indices


# ---> Implementations
def _preprocess(
        data: List[npt.NDArray], seq_shift: int, seq_len: int, n_segments: int
        ) -> npt.NDArray:
    '''
    Preprocess the data, and transform it to the desired form. More
    specifically, the features are concatenated to a single feature vectors,
    then segmented and lastly downsampled.

    Parameters
    ----------
    data: List[npt.NDArray]
        The original features. Each have shape (n_mels, n_windows)
    seq_shift: int
        The amount of shift between the segments
    seq_len: int
        The length of a single segment.
    n_segments: int
        The amount of segments to downsample to. (i.e. M from the paper)

    Returns
    -------
    npt.NDArray
        The transformed and downsampled features as a single array.
    '''

    # 1. Concatenate the data into single feature corpus
    # (has shape (total_samples, n_features))

    # Calculate the length of the created matrix
    assert len(data) > 0, "Input data contains no values!"
    total_len = sum(f.shape[1] for f in data)
    # Allocate the data
    # (todo: How to ensure that there is enough memory to allocate these arrays?)
    features = np.zeros((total_len, data[0].shape[0]))
    f_indices = np.zeros((total_len, 2))

    # Concatenate the values into single vectors, and create index arrays
    # that contain information on what samples correspond to which feature
    # vectors
    i = 0
    for k, fvec in enumerate(data):
        n = fvec.shape[1]
        features[i:i+n, :] = fvec.T
        f_indices[i:i+n, 0] = k
        f_indices[i:i+n, 1] = np.arange(fvec.shape[1])
        i += n
    # Remove any possible NaN or Inf values.
    features[np.isnan(features) | np.isinf(features)] = 0.0

    # 2. Split the features to fixed length segments
    # NOTE: if the division don't go exactly even, we just discard the
    # extra samples
    n_chunks = math.ceil((total_len - seq_len + 1)/seq_shift)
    x = np.zeros(combined_shape(n_chunks, seq_len, features.shape[1]))
    x_indices = np.zeros(combined_shape(n_chunks, seq_len, 2))

    # Create the segments, and maintain the original indexing
    loc = 0
    for i in range(n_chunks):
        x[i, ...] = features[loc:loc+seq_len, :]
        x_indices[i, ...] = f_indices[loc:loc+seq_len, :]
        loc += seq_shift

    # 3. Downsample the segments
    dim = x.shape[-1]
    bounds = np.rint(
            np.linspace(0, seq_len, num=n_segments+1)
            ).astype(np.uint64)

    x_downsampled = np.zeros((x.shape[0], n_segments*dim))
    for i in range(n_segments):
        tmp = np.mean(x[:, bounds[i]:bounds[i+1], :], axis=1)
        x_downsampled[:, (i*dim):((i+1)*dim)] = tmp

    return x_downsampled, x_indices


def _low_res_candidate_search_impl(
        ds_feats: npt.NDArray, original_shape: int, config: PDTWConfig
        ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    '''
    Run the low-resolution candidate search for the dataset

    Parameters
    ----------
    ds_feats: npt.NDArray
        The downsampled feature vectors
    original_shape: int
        The size of the original dataset.
    config: PDTWConfig
        The configuration for the run

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        Returns the distance probabilities, candidates the indices and the
        segments where the clips are silent.
    '''
    silent_idx = _voice_activity_detection(ds_feats, original_shape)
    print(f"Running for array of shape {ds_feats.shape}")
    distance_distr = _random_distances(
            ds_feats, config.n_nearest, config.max_ram, config.n_workers
    )

    print(f"Shape of distance-distribution: {distance_distr.shape}")
    # Fit the GMM model to the distance distribution
    idx = np.isnan(distance_distr)
    # Remove any nan values
    distance_distr = distance_distr[~idx]
    # make array 2D for the fitting (1 feature, multiple samples)
    distance_distr = distance_distr.reshape(-1, 1)

    # Calculate the mean and variance of the distance distribution
    # -> These will be considered as parameters for the normal distribution
    mu = distance_distr.mean()
    var = distance_distr.var()

    print("Starting to match low resolution candidates")
    # Match the lower candidates
    dist_probs, candidates = _find_n_nearest_segments(
            ds_feats, config.n_nearest, config.n_expansion, mu, var,
            config.alpha, config.max_ram, config.n_workers
    )

    print("Starting to remove dublicate pairs")
    # Remove possible duplicates (i.e. pairs that are in same order)
    candidates = _prune_candidates(candidates)
    return dist_probs, candidates, silent_idx


def _high_res_alignment_search_impl(x: npt.NDArray, indices: npt.NDArray):
    '''
    Perform the high-resolution candidate alignment algorithm
    '''
    # Take around 2 million samples from the distances
    all_dists = np.empty(2000000, dtype=np.float64)
    loc = 0
    while loc < all_dists.shape[0]:
        pass


def _voice_activity_detection(
        ds_feats: npt.NDArray, original_shape: int, confidence: float = 0.99
        ) -> npt.NDArray:
    '''
    Estimates the silent segments of the signal based on the energies
    of the signal

    Parameters
    ----------
    ds_feats: npt.NDArray
        The downsampled features vectors
    original_shape: int
        The original shape of the data
    confidence: float
        The probability threshold used to determine if the segment is silent or
        not. Should be between 0 and 1. Default 0.99
    Returns
    -------
    npt.NDArray
        The indices that are most likely silent.
    '''

    # Assume that the first coefficients are correlated with the signal
    # energies
    energies = np.sum(ds_feats[:, :ds_feats.shape[1]:original_shape], axis=1)
    energies = energies.reshape(-1, 1)
    # Imitate the settings of voicebox gmms
    model = GaussianMixture(
            n_components=2, covariance_type="diag",
            init_params="random_from_data", warm_start=True
    )

    # Fit the model until the classes have good enough representation
    while True:
        model.fit(energies)
        if model.weights_.min() > 0.05:
            break

    # Find out which class has more presentations
    idx = pairwise_distances_argmin(energies, model.means_, axis=1)
    idx1 = np.sum(idx == 0)
    idx2 = np.sum(idx == 1)
    silent_idx = 0 if idx1 < idx2 else 1
    # Assume that silent segments are contained in the class having less
    # activity
    prob_silent = 1 - stats.norm.cdf(
            energies, model.means_[silent_idx],
            np.sqrt(model.covariances_[silent_idx]),
    )*model.weights_

    # If the model is extremely sure that the section is silent, we take the
    # indexes
    return prob_silent > confidence


@nb.njit(
        cache=True, parallel=True, locals={
            'loc': nb.uint64, 'dist_distr': nb.float64[:, :],
            'chunk_size': nb.uint64, 'n_chunks': nb.uint64,
            "n_cols": nb.uint64, "dist_mtx": nb.float64[:, :]
        }
)
def _random_distances(
        ds_feats: npt.NDArray, n_nearest: int, max_ram: int, n_workers: int
        ) -> npt.NDArray:
    '''
    Samples the distances randomly to create "distance distribution"

    Parameters
    ----------
    ds_feats: npt.NDArray
        The downsampled feature vectors.
    n_nearest: int
        The amount of samples to select from the distance matrix during each
        iteration
    max_ram: int
        The maximum amount of RAM to use to store the values at a single time
        (in bytes).
    n_workers: int
        The amount of workers used to run the parallized code TODO: remove this
        and use numbas api to get the amount of workers.
    Returns
    -------
    npt.NDArray
        The sampling of distances from the pairwise distace matrices
    '''
    # NOTE: The chunking is done approximately to respect user boundaries, but
    # in such way that the output can be shaped similarly

    # The amount of samples the array is going to have, given the RAM
    # constraints placed by the user
    n_samples = math.floor(max_ram/ds_feats.itemsize/n_workers)
    chunk_size = math.floor(n_samples/ds_feats.shape[0])
    n_chunks = math.floor(ds_feats.shape[0]/chunk_size)

    # Assuming here that the split went evenly
    dist_distr = np.zeros((n_chunks, ds_feats.shape[0]*n_nearest))

    # Iterate over each "chunk" of the data.
    # Writing to dist_distr is safe, as we are never assigning to same index
    # from same coordinates
    for chunk in nb.prange(n_chunks):
        loc = chunk*chunk_size
        endpoint = min(loc + chunk_size, ds_feats.shape[0])
        dist_mtx = pdist_cos(ds_feats[loc:endpoint, :], ds_feats)
        # NOTE: np.random.Generator is not thread safe with numba -> use old
        # style random number generation. It should suffice for now.

        # Select the samples to check at random
        # TODO: Ensure that these are actually generated randomly for each
        # iteration
        idx = np.random.randint(
                0, high=ds_feats.size, size=ds_feats.shape[0]*n_nearest
        )

        # NOTE: Numba doesn't currently support np.unravel_index nor indexing
        # by multiple numpy arrays at the sametime, and thus the copying must
        # be done manually.
        n_cols = dist_mtx.shape[1]
        for ii in nb.prange(idx.shape[0]):
            lin_index = idx[ii]
            xi, yi = lin_index // n_cols, lin_index % n_cols
            dist_distr[chunk, ii] = dist_mtx[xi, yi]
    return dist_distr


@nb.njit(parallel=True)
def _find_n_nearest_segments(
        x: npt.NDArray, n_nearest: int, n_lap_frames: int,
        mu: float, var: float, alpha: float, max_ram: int,
        n_workers: int
        ) -> Tuple[npt.NDArray, npt.NDArray]:
    '''
    Finds the n-nearest segments using probabilistic measures. Corresponds to
    end portion of the the low-resolution candidate search in the paper

    Parameters
    ----------
    x: npt.NDArray
        The features
    n_nearest: int
        The amount of nearest neighbours to consider for each segment.
    n_lap_frames: int
        How many neighbouring segments are discarded from the matching.
    mu: float
        The mean value extracted from the fitted GMM (There should be exactly
        one as there is single observation)
    var: float
        The covariance value extracted from the fitted GMM. (There should be
        exactky one as there is single observation)
    alpha: float
        The required probability for each match to be considered.
    max_ram: int
        The maximum amount of ram (in bytes) that can be allocated for a single
        distance array.
    n_workers: int
        The amount of workers used to calculate the results.
    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        The distance probabilities of the candidates, and the candidate indices
    '''
    dist_probs = np.empty((x.shape[0], n_nearest))
    candidate_indices = np.zeros((x.shape[0], n_nearest), dtype=np.uint64)

    # Calculate the amount of samples per worker, and then divide that by
    # the amount of samples to get the chunk-size for the data
    chunksize = math.floor(max_ram/x.itemsize/n_workers/x.shape[0])
    n_chunks = math.ceil(x.shape[0]/chunksize)

    loc = 1
    for chunk in nb.prange(n_chunks):
        endpoint = min(loc + chunksize, x.shape[0])

        # Calculate the pairwise distances in chunks
        dist_near = pdist_cos(x[loc:endpoint, :], x)

        # Ensure that neighbouring segments are not matched!
        for i in nb.prange(-n_lap_frames, n_lap_frames):
            for ii in nb.prange(dist_near.shape[0]):
                idx = loc + i + ii
                if 0 <= idx <= dist_near.shape[0]:
                    dist_near[ii, loc+i+ii] = np.nan

        # NOTE: Numba doesn't support axis keyword for the argsort, so we must
        # manually run the sort along the last-axis.
        # Also, argpartition would be enough here, but it is also un-supported
        # currently
        ind_near = np.empty(dist_near.shape, dtype=np.uint64)
        for i in range(ind_near.shape[0]):
            ind_near[i, :] = np.argsort(dist_near[i, :])

        # Again, 2D indexing is not allowed, so the sorting of the rows must
        # be done manually.
        tmp = np.empty_like(dist_near)
        for i in range(ind_near.shape[0]):
            for ii in range(ind_near.shape[1]):
                tmp[i, ii] = dist_near[i, ind_near[i, ii]]
        dist_near = tmp

        dist_near = dist_near[:, :n_nearest]
        ind_near = ind_near[:, :n_nearest]
        # Mark the neighbouring frames (i.e. Array values with NaN) to the
        # indexing table
        # Again, the fancy indexing is not supported, so do it manually
        for i in range(ind_near.shape[0]):
            for ii in range(ind_near.shape[1]):
                if np.isnan(dist_near[i, ii]):
                    ind_near[i, ii] = _IS_NEIGHBOUR

        # Interpret the sampling of the distance distribution as normally
        # distributed, and calculate the cdf values for it.
        dist_near = np.ascontiguousarray(dist_near)
        probs = norm_cdf(dist_near, mu, var)

        # Now, select only those canditates that are probable enough
        for i in range(dist_near.shape[0]):
            for ii in range(dist_near.shape[1]):
                if probs[i, ii] > alpha:
                    dist_near[i, ii] = np.nan
                    ind_near[i, ii] = _IS_NEIGHBOUR

        # Store the candidates
        dist_probs[loc:endpoint, :] = dist_near
        candidate_indices[loc:endpoint, :] = ind_near

        loc += chunksize
    return dist_probs, candidate_indices


@nb.njit
def _sample_random_distances(
        dist_probs: npt.NDArray, candidate_indices: npt.NDArray, n_samples: int
        ) -> npt.NDArray:
    '''
    Sample randomly from the distance distribution to calculate the pair-wise
    statistics of the frame-level distances.

    Parameters
    ----------
    dist_probs: npt.NDArray
        The distribution of the distances.
    candidate_indices: npt.NDArray
        The indexes of the candidate pairs.
    n_samples: int
        The amount of samples to pick from the distribution. Should be in range
        of 10^6
    '''
    # out = np.empty(n_samples, dtype=np.float64)
    # loc = 0
    # while loc < n_samples:
    #     k = np.random.randint(0, candidate_indices.shape[0])
    #     a = candidate_indices[k, :, 1]
    #     b = candidate_indices[k, :, 2]
    #     most_common = mode(a)
    #
    #     # Create new array, withtout the most-common values
    #     # Remove most common values from b
    # pass


@nb.njit(nb.uint64[:, :](nb.uint64[:, :]))
def _prune_candidates(indexes: npt.NDArray) -> npt.NDArray:
    '''
    Removes possible duplicates (i.e. pairs that appear multiple times in
    in different order from the dataset)

    Parameters
    ----------
    indexes: npt.NDArray
        The indexes of the candidates
    '''
    for i in range(indexes.shape[0]):
        for j in range(indexes.shape[1]):
            val = indexes[i, j]
            if val == _IS_NEIGHBOUR:
                continue
            # Find if the array has the same pairs, but the other-way around
            for k in range(indexes.shape[1]):
                val2 = indexes[val, k]
                if val2 == i:
                    indexes[val, k] = _IS_NEIGHBOUR
    return indexes
