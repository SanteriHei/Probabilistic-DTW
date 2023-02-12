from meta_types import Path

import math
import pathlib

import numpy as np
import librosa as lb
from tqdm import tqdm

import numpy.typing as npt
from typing import List, Tuple


def get_features(
        dirpath: Path, win_length: float = 0.025,
        hop_length: float = 0.0125, target_sr: int = 8000, n_mfcc: int = 13
        ) -> List[Tuple[str, npt.NDArray]]:
    '''
    Calculates the features from the given wav-files. By default calculates
    the MFCC's, deltas and delta deltas and normalizes them.

    Parameters
    ----------
    dirpath: Path
        Path to the directory containing the audio files that should be read.
        NOTE: reads only *.wav files from the directory
    win_length: float, optional
        The window length, expressed as a fraction of the target sample-rate.
        Default 0.025
    hop_length: float, optional
        The step between each frame, expressed as a fraction of the target
        sample-rate. Default 0.0125
    target_sr: int, optional
        The used sampling rate for the signals. Default 8000
    n_mfcc: int, optional
        The amount of mfcc's to calculate from the signal. Default 13.

    Returns
    -------
    List[npt.NDArray]
        List of feature arrays, each having shape
        (3*n_mfcc, len(audio)/(win_length - hop_lenght) + 1)

    '''
    dirpath = pathlib.Path(dirpath)
    # Convert the window length and shift to samples
    hop_length = math.floor(hop_length*target_sr)
    wl = math.floor(win_length*target_sr)
    out = []
    for fp in tqdm(dirpath.glob("*.wav")):
        # Load, and resample the data
        y, sr = lb.load(fp, sr=target_sr)
        n_feats = int(y.shape[0]/(wl - hop_length) + 1)
        feats = np.zeros((n_mfcc*3, n_feats))

        # Calculate the MFCC's
        mfcc = lb.feature.mfcc(
                y=y, sr=sr, hop_length=hop_length, win_length=wl,
                window="hamming", n_mfcc=n_mfcc
        )

        # Calculate the deltas
        delta = lb.feature.delta(mfcc, order=1)
        delta2 = lb.feature.delta(mfcc, order=2)

        feats[:n_mfcc, :] = mfcc
        feats[n_mfcc:2*n_mfcc, :] = delta
        feats[2*n_mfcc:, :] = delta2

        # Normalize the features
        feats -= np.nanmean(feats, axis=0)
        feats /= np.nanstd(feats, axis=0)
        out.append((fp.name, feats))
    return out
