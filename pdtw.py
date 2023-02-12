from utils import combined_shape

import math
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin
import numba as nb
from numba_stats import norm

from typing import List, Tuple
import numpy.typing as npt


@dataclass
class PDTWConfig:
    alpha: float
    n_nearest: int
    seq_shift: int
    n_slices: int
    seq_len: int
    n_expansion: int


def pdtw(data: List[npt.NDArray], config: PDTWConfig):
    features, energies = _preprocess(
            data, config.seq_shift, config.seq_len, config.n_segments
    )

    _canditate_search(features, energies)
    pass


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
        The original features
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

    # Calculate the length of the created matrix
    total_len = sum(max(f.shape) for f in data)
    n = data[0].shape[0]

    # 1. Compress data to a single feature vector
    features = np.zeros((total_len, data[0].shape[1]))
    f_indices = np.zeros((total_len, 2))

    it = range(0, len(data), n)
    for i, mtx in zip((it, data)):
        features[i:i+n, :] = mtx
        f_indices[i:i+n, 1] = i
        f_indices[i:i+n, 2] = np.arange(max.shape[0])

    # Remove any possible NaN or Inf values.
    features[np.isnan(features) | np.isinf(features)] = 0.0

    # 2. Split the features to fixed length segments
    shape = combined_shape(
            math.floor(total_len/seq_shift) - 10, seq_len, features.shape[1]
    )
    x = np.zeros(shape)
    shape = combined_shape(math.floor(total_len/seq_shift) - 10, seq_len, 2)
    x_indices = np.zeros(shape)

    for i, loc in enumerate(range(0, features.shape[0]-seq_len, step=seq_shift)):
        x[i, ...] = features[loc:loc+seq_len, :]  # The segments
        x_indices[i, ...] = f_indices[loc:loc+seq_len, :]  # Maintain indexing

    # 3. Downsample the segments
    dim = x.shape[-1]
    bounds = np.linspace(0, seq_len, num=n_segments)

    x_downsampled = np.zeros((x.shape[1], n_segments*dim))
    for i in range(n_segments):
        tmp = np.mean(x[:, bounds[i]:bounds[i+1], :], axis=1)
        x_downsampled[:, ] = tmp

    # Also extract the first coefficients, assuming that they are correlated
    # with the signal energy
    first_coeff_idx = np.arange(0, x_downsampled.shape[1], features.shape[1])
    energies = np.sum(x_downsampled[:, first_coeff_idx], axis=1)

    return x_downsampled, energies


def _canditate_search(features, energies):
    '''
    Run the low-resolution candidate search for the dataset
    '''
    silent_idx = _voice_activity_detection(energies)


def _voice_activity_detection(energies):
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
            energies, model.weights_[silent_idx],
            np.sqrt(model.covariances_[silent_idx]),
    )*model.weights_

    # If the model is extremely sure that the section is silent, we take the
    # indexes
    return prob_silent > 0.99


@nb.njit
def _pdist_cos(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    '''
    Calculate the cosine pairwise distance between two vectors
    '''
    w = x.shape[0]
    h = y.shape[0]
    out = np.zeros((w, h), dtype=x.dtype)
    for i in range(w):
        for j in range(h):
            out[i, j] = np.dot(x[i], y[j])/(np.sqrt(np.sum(x[i]**2))*np.sqrt(np.sum(y[i]**2)))
    return out


@nb.njit
def _pdist_euc(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    w = x.shape[0]
    h = y.shape[0]
    out = np.zeros((w, h), dtype=x.dtype)
    for i in range(w):
        for j in range(h):
            out[i, j] = np.sqrt(np.sum((x[i] - y[i])**2))
    return out


@nb.jit(cache=True, parallel=True)
def _random_distances(x: npt.NDArray):
    # TODO: Lines 217 - 268 from the matlab code
    chunk_size = math.floor(32/x.shape[0])
    n_chunks = math.floor(x.shape[0]/chunk_size)
    loc = 0
    for chunk in nb.prange(n_chunks):
        endpoint = min(loc + chunk_size, x.shape[0])
        tmp = _pdist_cos(x[loc:endpoint, :], x)


    pass


@nb.njit(cache=True, parallel=True)
def _find_n_nearest_segments(
        x: npt.NDArray, n_nearest: int, n_lap_frames: int,
        mu: npt.NDArray, var: npt.NDArray, alpha: float
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

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        The distance probabilities of the candidates, and the candidate indices
    '''
    dist_probs = np.empty((x.shape[0], n_nearest))
    candidate_indices = np.empty((x.shape[0], n_nearest))
    chunksize = math.floor(32/x.shape[0])
    n_chunks = math.ceil(x.shape[0]/chunksize)
    loc = 1
    for chunk in nb.prange(n_chunks):
        endpoint = min(loc + chunksize, x.shape[0])

        # Calculate the pairwise distances in chunks?
        dist_near = _pdist_cos(x[loc:endpoint, :], x)

        # Ensure that neighbouring segments are not matched!
        for i in nb.prange(-n_lap_frames, n_lap_frames):
            for ii in nb.prange(dist_near.shape[0]):
                idx = loc + i + ii
                if 0 <= idx <= dist_near.shape[0]:
                    dist_near[ii, loc+i+ii] = np.nan

        # Sort only the first n_nearest elements that we are looking for.
        # This is only O(n) operation compared to O(n*log(n)) for full sort
        ind_near = np.argpartition(dist_near, n_nearest, axis=-1)
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


def _alignment_search(x: npt.NDArray):
    pass
