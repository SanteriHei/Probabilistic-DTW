from feature_extraction import get_features
from pdtw import PDTWConfig, low_res_search
import utils

import datetime
import pathlib
import multiprocessing as mp
import argparse

import numpy as np

_TS_FORMAT: str = "%H-%M-%ST%d-%m-%Y"


def _extract_features(args: argparse.Namespace):
    dirpath = pathlib.Path(args.dirpath)

    if not dirpath.exists() or not dirpath.is_dir():
        raise FileNotFoundError((f"{str(dirpath)!r} does not point to a valid "
                                "directory"))

    features = get_features(dirpath, args.win_length, args.hop_length,
                            args.target_sr, args.n_mfcc)

    # Ensure that the output directory exist
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for fname, feats in features:
        np.save((outdir / fname).with_suffix(".npy"), feats)


def _low_res_search(args: argparse.Namespace):
    '''
    Runs the low-resolution candidate search from the paper, and stores the
    results to a given directory.

    Parameters
    ----------
    args: argparse.Namespace
    '''
    config = PDTWConfig(
            alpha=args.alpha, n_nearest=args.n_nearest,
            seq_shift=args.seq_shift, n_frames=args.n_frames,
            seq_len=args.seq_len, n_expansion=0,  # Not needed in this search
            max_ram=args.max_ram, n_workers=args.n_workers,
            verbose=args.verbose
    )

    features = utils.load_features(args.feature_path)
    ds_feats, ds_indices, dist_probs, candidates, silent_idx = low_res_search(
            features, config
    )

    if config.verbose:
        print(f"Found {candidates.shape[0]} candidates")

    dir_path = pathlib.Path(args.output_path)

    # Ensure that the directory exists
    if dir_path.is_file():
        raise FileExistsError((f"{str(dir_path)!r} points to an existing file "
                               "It should be a path to a directory"))
    dir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime(_TS_FORMAT)
    fp = dir_path / f"candidates_{ts}.npz"
    np.savez(
            fp, ds_feats=ds_feats, ds_indices=ds_indices,
            dist_probs=dist_probs, candidates=candidates,
            silent_idx=silent_idx
    )
    print(f"Saved candidate to {str(fp)!r}")


def _get_parser():
    parser = argparse.ArgumentParser()

    # ========== Feature extraction =============
    sub_parser = parser.add_subparsers(help="sub-commands")
    feat_extraction = sub_parser.add_parser(
            "extract-features", help="Extracts features from a given dataset"
    )

    feat_extraction.add_argument(
            "dirpath", type=str, help="Path to the dataset"
    )
    feat_extraction.add_argument(
            "outdir", type=str, help=("Path to directory, where the extracted "
                                      "features will be saved to")
    )
    feat_extraction.set_defaults(func=_extract_features)

    # ========== Tunable parameters ===========
    feat_extraction_param_group = feat_extraction.add_argument_group(
            "Parameters"
    )

    feat_extraction_param_group.add_argument(
            "--win-length", type=float, default=0.025, dest="win_length",
            help=("Defines the window size. Expressed as the fraction of the "
                  "desired sampling rate. Default %(default)s")
    )

    feat_extraction_param_group.add_argument(
            "--hop-length", type=float, default=0.0125, dest="hop_length",
            help=("Defines the step between frames. Expressed as the fraction "
                  "of the desired sampling rate. Default %(default)s")
    )

    feat_extraction_param_group.add_argument(
            "--target-sr", type=int, default=8000,  dest="target_sr",
            help=("Defines the used sampling-rate. Default %(default)s")
    )

    feat_extraction_param_group.add_argument(
            "--n-mfcc", type=int, default=13, dest="n_mfcc",
            help="Defines the amount of MFCC's extracted. Default %(default)s"
    )

    # ======= Low-resolution candidate search ==============
    low_res_search_parser = sub_parser.add_parser(
            "low-res-candidate-search",
            help="Run the low-resolution candidate search"
    )

    low_res_search_parser.add_argument(
            "feature_path", type=str, metavar="feature-path",
            help=("Path to the directory containing the pre-calculated "
                  "feature vectors")
    )

    low_res_search_parser.add_argument(
            "output_path", type=str, metavar="output-path",
            help=("Path to the directory where the results will be stored to")
    )

    low_res_search_parser.add_argument(
            "-v", "--verbose", action="store_true",
            help=("If this flag is set, additional information is printed out "
                  "during the process")
    )

    # ======== Tunable parameters ============
    search_param_group = low_res_search_parser.add_argument_group("Parameters")

    search_param_group.add_argument(
            "--alpha", type=float, default=0.001,
            help=("Defines the signicance criterion of selecting the "
                  "interesting segments. Can be seen as the maximum "
                  "probability that a given distance is not smaller than the "
                  "usual distances in the corpus. Default %(default)s")
    )

    search_param_group.add_argument(
            "--n-nearest", type=int, default=5, dest="n_nearest",
            help=("The number of nearest neighbours that are considered for "
                  "each segment. Corresponds to 'k' in the paper. Default "
                  "%(default)s")
    )

    search_param_group.add_argument(
            "--seq-shift", type=int, default=10, dest="seq_shift",
            help=("The shift between the segments. Corresponds to 'S' in the "
                  "paper. Default %(default)s")
    )

    search_param_group.add_argument(
            "--seq-len", type=int, default=20, dest="seq_len",
            help=("The length of segments used during the low-resolution "
                  "matching. Corresponds to 'L' in the paper. "
                  "Default %(default)s")
    )

    search_param_group.add_argument(
            "--n-frames", type=int, default=4, dest="n_frames",
            help=("The number of frames in the downsampled segments. "
                  "Corresponds to 'M' in the paper. Default %(default)s")
    )

    comp_detail_group = low_res_search_parser.add_argument_group(
            "Computational details"
    )
    comp_detail_group.add_argument(
            "--max-ram", type=int, default=int(8e9), dest="max_ram",
            help=("The maximum amount of ram used for the distance metric "
                  "calculations (in bytes). Default %(default)s")
    )

    comp_detail_group.add_argument(
            "--n-workers", type=int, default=mp.cpu_count(), dest="n_workers",
            help=("The amount of workers used to run the parallized regions. "
                  "Default %(default)s")
    )

    low_res_search_parser.set_defaults(func=_low_res_search)

    # If not command is given, just print out the help
    parser.set_defaults(
            func=lambda args: args.parser.print_help(), parser=parser
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    args.func(args)
