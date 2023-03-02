from utils import combined_shape, pdist_cos, pdist_euc, get_chunksize

import math
import pathlib
import datetime
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin
import numba as nb
from numba_stats import norm

from typing import List, Tuple
import numpy.typing as npt


# Format for the timestamps
_TS_FORMAT: str = "%H-%M-%ST%d-%m-%Y"


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


def low_res_search(features: List[npt.NDArray], config: PDTWConfig):
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
    Tuple[npt.NDArray, npt.NDArray]
        Returns the distance probabilities and the candidates from the search
    '''
    ds_feats = _preprocess(
            features, config.seq_shift, config.seq_len, config.n_frames
    )

    nb.set_num_threads(config.n_workers)
    return _low_res_candidate_search_impl(ds_feats, features[0].shape[0])


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

    return x_downsampled


def _low_res_candidate_search_impl(
        ds_feats: npt.NDArray, original_shape: int
        ) -> Tuple[npt.NDArray, npt.NDArray]:
    '''
    Run the low-resolution candidate search for the dataset

    Parameters
    ----------
    ds_feats: npt.NDArray
        The downsampled feature vectors
    original_shape: int
        The size of the original dataset.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        The distance probabilities and the candidate segment indexes
    '''
    silent_idx = _voice_activity_detection(ds_feats, original_shape)
    print(silent_idx)
    distance_distr = _random_distances(ds_feats)

    # Fit the GMM model to the distance distribution
    idx = np.isnan(distance_distr)
    # Remove any nan values
    distance_distr = distance_distr[~idx]
    model = GaussianMixture(
            n_components=1, covariance_type="diag",
            init_params="random_from_data", warm_start=True
    )
    model.fit(distance_distr)

    # sort the model properties
    means = model.means_.copy()
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    vars = model.covariances_.copy()[sort_idx]
    weights = model.weights_.copy()[sort_idx]

    # Match the lower candidates
    dist_probs, candidates = _find_n_nearest_segments(ds_feats)
    return dist_probs, candidates


def _high_res_alignment_search_impl(x: npt.NDArray):
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
            'loc': nb.uint64, 'i': nb.uint64, 'dist_distr': nb.float64[:, :],
            'chunk_size': nb.uint64, 'n_chunks': nb.uint64
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

    Returns
    -------
    npt.NDArray
        The sampling of distances from the pairwise distace matrices
    '''
    # NOTE: The chunking is done approximately to respect user boundaries, but
    # in such way that the output can be shaped similarly

    # float64 -> 8 bytes per element
    n_samples = math.floor(max_ram/8/n_workers)
    chunk_size = math.floor(n_samples/ds_feats.shape[0])
    n_chunks = math.floor(ds_feats.shape[0]/chunk_size)

    # Assuming here that the split went evenly
    dist_distr = np.zeros((n_chunks, ds_feats.shape[0]*n_nearest))

    loc = 0
    i = 0
    for chunk in nb.prange(n_chunks):
        endpoint = min(loc + chunk_size, ds_feats.shape[0])
        dist_mtx = pdist_cos(ds_feats[loc:endpoint, :], ds_feats)

        # NOTE: np.random.Generator is not thread safe with numba -> use old
        # style random number generation. It should suffice for now.

        # Select the samples to check at random
        idx = np.random.randint(
                0, high=ds_feats.size, size=ds_feats.shape[0]*n_nearest
        )
        xi, yi = np.unravel_index(idx, dist_mtx.shape)
        dist_distr[i, :] = dist_mtx[xi, yi]
        i += 1
    return dist_distr


@nb.njit(cache=True, parallel=True)
def _find_n_nearest_segments(
        x: npt.NDArray, n_nearest: int, n_lap_frames: int,
        mu: npt.NDArray, var: npt.NDArray, alpha: float, max_ram: int,
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
    mu: npt.NDArray
        The mean values from the
    var: npt.NDArray
        The covariance values extracted from the fitted GMM.
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
    candidate_indices = np.empty((x.shape[0], n_nearest))

    # using 64 bit floats -> 8 bytes per element.
    chunksize = math.floor(max_ram/8/n_workers/x.shape[0])
    n_chunks = math.ceil(x.shape[0]/chunksize)
    loc = 1
    for chunk in nb.prange(n_chunks):
        endpoint = min(loc + chunksize, x.shape[0])

        # Calculate the pairwise distances in chunks?
        dist_near = pdist_cos(x[loc:endpoint, :], x)

        # Ensure that neighbouring segments are not matched!
        for i in nb.prange(-n_lap_frames, n_lap_frames):
            for ii in nb.prange(dist_near.shape[0]):
                idx = loc + i + ii
                if 0 <= idx <= dist_near.shape[0]:
                    dist_near[ii, loc+i+ii] = np.nan

        # Sort only the first n_nearest elements that we are looking for.
        # This is only O(n) operation compared to O(n*log(n)) for full sort

        # TODO: !!! argpartition is not supported with parallel true!!!
        ind_near = np.argsort(dist_near, axis=-1)[:n_nearest]
        dist_near = dist_near[ind_near]
        dist_near = dist_near[:n_nearest, :]
        ind_near = ind_near[:n_nearest, :]

        ind_near[np.isnan(dist_near)] = np.nan

        # Convert the distances to probabilities, that measure how likely it is
        # that such small distances are encoutered in the corpus
        # NOTE: using numba_stats here, as scipy.stats is not numba aware

        probs = norm.cdf(dist_near, mu[0], var[0])
        # Now, select only those canditates that are probable enough
        dist_near[probs > alpha] = np.nan
        ind_near[probs > alpha] = np.nan

        # Store the candidates
        dist_probs[loc:endpoint, :] = dist_near
        candidate_indices[loc:endpoint, :] = ind_near

        loc += chunksize

    return dist_probs, candidate_indices
