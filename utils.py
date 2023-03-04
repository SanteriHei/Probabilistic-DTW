''' This module defines some helper functions and jitted distance functions'''
from meta_types import Path

import pathlib
import math

import numpy as np
import numba as nb


from typing import Tuple, Union, List
import numpy.typing as npt


def combined_shape(
        dim1: Union[Tuple[int, ...], int], *args: Tuple[int, ...]
        ) -> Tuple[int, ...]:
    '''
    dim1: Tuple[int, ...] | int
        The 'base' dimension that should be merged with the others
    args: Tuple[int, ...]
        The other dimensions that should be merged

    Returns
    -------
    Tuple[int, ...]
        One single combined shape that consisting of all the given shapes
    '''
    if isinstance(dim1, tuple):
        return (*dim1, *args)
    return (dim1, *args)


def get_chunksize(
        max_ram: int, n_workers: int, *, bytes_per_elem: int, total_size: int
        ) -> int:
    '''
    Calculates the approximate estimate for the chunk-size for the array, given
    the amount of workers and ram.

    Parameters
    ----------
    max_ram: int
        The amount of ram to be used (in bytes).
    n_workers: int
        The amount of workers used to process the elements
    bytes_per_elem: int
        The amount of bytes each processed items contains.
    total_size: int
        The total size of the processed array

    Returns
    -------
    int
        The (conservative) estimate on the chunk-size given the settings. If
        the array is not divisidable to equally sized chunks, this result won't
        be accurate.
    '''
    n_samples = max_ram/bytes_per_elem/n_workers
    return math.floor(n_samples/total_size)


def load_features(dir_path: Path) -> List[npt.NDArray]:
    '''
    Loads the pre-extracted features from a given directory.

    Parameters
    ----------
    dir_path: Path
        Path to the directory containing the extraced features
    '''
    dir_path = pathlib.Path(dir_path)

    if not dir_path.is_dir() or not dir_path.exists():
        raise FileNotFoundError((f"{str(dir_path)!r} does not point to a "
                                 "valid directory!"))
    return [np.load(fp) for fp in dir_path.glob("*.npy")]


@nb.njit
def index_to_coords(
        index: npt.NDArray, shape: Tuple[int, int]
        ) -> Tuple[npt.NDArray, npt.NDArray]:
    '''
    Convert the linear indexes to 2D coordinates
    (Similarly to what np.unravel_index does).

    Parameters
    ----------
    indexes: npt.NDArray
        List of linear indexes to a 2d numpy array
    shape: Tuple[int, int]
        The shape of the target array

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        The corresponding 2D indexes (x first, y second)
    '''
    xs = np.empty(index.size)
    ys = np.empty(index.size)
    for ii, idx in enumerate(index):
        xs[ii] = idx // shape[1]
        ys[ii] = idx % shape[1]
    return xs, ys


@nb.njit(
        nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]),
        locals={'tmp1': nb.float64, 'tmp2': nb.float64}
)
def pdist_cos(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    '''
    Calculates the cosine pairwise distance between two vectors.

    Here the cosine distance is defined as 1 - cos-similarity(x, y)

    Parameters
    ----------
    x: npt.NDArray
        The other vector. Should have shape (x_rows, n_cols)
    y: npt.NDArray
        The other vector. Should have shape (y_rows, n_cols)

    Returns
    -------
    npt.NDArray
        The distance matrix between the vectors. Has shape (x_rows, y_rows)
        where the index [i, j] correponds to the distance between ith row of x
        and jth row of y
    '''
    w = x.shape[0]
    h = y.shape[0]
    out = np.empty((w, h), dtype=x.dtype)
    for i in range(w):
        for j in range(h):
            tmp1 = np.sqrt(np.sum(x[i, :]**2))
            tmp2 = np.sqrt(np.sum(y[j, :]**2))
            out[i, j] = 1 - np.dot(x[i, :], y[j, :])/(tmp1*tmp2)
    return out


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]))
def pdist_euc(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    '''
    Calculates the pairwise euclidian distance between two vectors.

    Parameters
    ----------
    x: npt.NDArray
        The other vector. Should have shape (x_rows, n_cols)
    y: npt.NDArray
        The other vector. Should have shape (y_rows, n_cols)

    Returns
    -------
    npt.NDArray
        The distance matrix between the vectors. Has shape (x_rows, y_rows)
        where the index [i, j] corresponds to the distance between ith row of x
        and jth row of y.
    '''
    w = x.shape[0]
    h = y.shape[0]
    out = np.empty((w, h), dtype=x.dtype)
    for i in range(w):
        for j in range(h):
            out[i, j] = np.sqrt(np.sum((x[i, :] - y[j, :])**2))
    return out
