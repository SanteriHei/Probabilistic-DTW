from feature_extraction import get_features

import pathlib
import argparse

import numpy as np


def _extract_features(args: argparse.Namespace):
    dirpath = pathlib.Path(args.dirpath)

    if not dirpath.exists() or not dirpath.is_dir():
        raise FileNotFoundError((f"{str(dirpath)!r} does not point to a valid "
                                "directory"))

    features = get_features(dirpath, args.win_lenght, args.hop_lenght,
                            args.target_sr, args.n_mfcc)

    # Ensure that the output directory exist
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for fname, feats in features:
        np.save((outdir / fname).with_suffix(".npy"), feats)


def _get_parser():
    parser = argparse.ArgumentParser()

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
    parameter_group = feat_extraction.add_argument_group("Parameters")

    parameter_group.add_argument(
            "--win-length", type=float, default=0.025, dest="win_lenght", 
            help=("Defines the window size. Expressed as the fraction of the "
                  "desired sampling rate. Default %(default)s")
    )

    parameter_group.add_argument(
            "--hop-lenght", type=float, default=0.0125, dest="hop_lenght",
            help=("Defines the step between frames. Expressed as the fraction "
                  "of the desired sampling rate. Default %(default)s")
    )

    parameter_group.add_argument(
            "--target-sr", type=int, default=8000,  dest="target_sr",
            help=("Defines the used sampling-rate. Default %(default)s")
    )

    parameter_group.add_argument(
            "--n-mfcc", type=int, default=13, dest="n_mfcc",
            help="Defines the amount of MFCC's extracted. Default %(default)s"
    )

    # If not command is given, just print out the help
    parser.set_defaults(
            func=lambda args: args.parser.print_help(), parser=parser
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    args.func(args)
