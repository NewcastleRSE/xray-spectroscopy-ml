"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

from pathlib import Path

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################


def unique_path(path: Path, base_name: str) -> Path:
    # returns a unique path from `p`/`base_name`_001, `p`/`base_name`_002,
    # `p`/`base_name`_003, etc.

    n = 0
    while True:
        n += 1
        unique_path = path / (base_name + f"_{n:03d}")
        if not unique_path.exists():
            return unique_path


def linecount(f: Path) -> int:
    # returns the linecount for a file (`f`)

    with open(f, "r") as f_:
        return len([l for l in f_])


def list_files(d: Path, with_ext: bool = True) -> list:
    # returns a list of files (as POSIX paths) found in a directory (`d`);
    # 'hidden' files are always omitted and, if with_ext == False, file
    # extensions are also omitted

    return [
        (f if with_ext else f.with_suffix(""))
        for f in d.iterdir()
        if f.is_file() and not f.stem.startswith(".")
    ]


def list_filestems(d: Path) -> list:
    # returns a list of file stems (as strings) found in a directory (`d`);
    # 'hidden' files are always omitted

    return [f.stem for f in list_files(d)]


def str_to_numeric(str_: str):
    # returns the numeric (floating-point or integer) cast of `str_` if
    # cast is allowed, otherwise returns `str_`

    try:
        return float(str_) if "." in str_ else int(str_)
    except ValueError:
        return str_


def print_nested_dict(dict_: dict, nested_level: int = 0):
    # prints the key:value pairs in a dictionary (`dict`) in the format
    # '>> key :: value'; iterates recursively through any subdictionaries,
    # indenting with two white spaces for each sublevel (`nested level`)

    for key, val in dict_.items():
        if not isinstance(val, dict):
            if isinstance(val, list):
                val = f"[{val[0]}, ..., {val[-1]}]"
            print("  " * nested_level + f">> {key} :: {val}")
        else:
            print("  " * nested_level + f">> {key}")
            print_nested_dict(val, nested_level=nested_level + 1)

    return 0


def print_cross_validation_scores(scores: dict, model_mode: str):
    # prints a summary table of the scores from k-fold cross validation;
    # summarises the elapsed time and train/test metric scores for each k-fold
    # with overall k-fold cross validation statistics (mean and std. dev.)
    # using the `scores` dictionary returned from `cross_validate`

    # print(scores)
    print("")
    print(">> summarising scores from k-fold cross validation...")
    print("")

    if model_mode == "mlp" or model_mode == "cnn":
        #
        print("*" * 16 * 3)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test"))
        print("*" * 16 * 3)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}"
        for kf, (t, train, test) in enumerate(
            zip(scores["fit_time"], scores["train_score"], scores["test_score"])
        ):
            print(fmt.format(kf, t, np.absolute(train), np.absolute(test)))

        print("*" * 16 * 3)
        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in ("fit_time", "train_score", "test_score")
        )
        print(fmt.format("std. dev.", *stdevs_))

        print("*" * 16 * 3)

        print("")

    if model_mode == "ae_mlp" or model_mode == "ae_cnn":
        #
        print("*" * 16 * 4)
        fmt = "{:<10s}{:>6s}{:>16s}{:>16s}{:>16s}"
        print(fmt.format("k-fold", "time", "train", "test recon", "test pred"))
        print("*" * 16 * 4)

        fmt = "{:<10.0f}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        for kf, (t, train, test_recon, test_pred) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_score"],
                scores["test_pred_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(test_recon),
                    np.absolute(test_pred),
                )
            )

        print("*" * 16 * 4)

        fmt = "{:<10s}{:>5.1f}s{:>16.8f}{:>16.8f}{:>16.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_score",
                "test_pred_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))

        print("*" * 16 * 4)

        print("")

    if model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
        #
        print("*" * (16 + 18 * 5))
        fmt = "{:<10s}{:>6s}{:>18s}{:>18s}{:>18s}{:>18s}{:>18s}"
        print(
            fmt.format(
                "k-fold",
                "time",
                "train",
                "test recon xyz",
                "test recon xanes",
                "test pred xyz",
                "test pred xanes",
            )
        )
        print("*" * (16 + 18 * 5))

        fmt = "{:<10.0f}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        for kf, (t, train, recon_xyz, recon_xanes, pred_xyz, pred_xanes) in enumerate(
            zip(
                scores["fit_time"],
                scores["train_score"],
                scores["test_recon_xyz_score"],
                scores["test_recon_xanes_score"],
                scores["test_pred_xyz_score"],
                scores["test_pred_xanes_score"],
            )
        ):
            print(
                fmt.format(
                    kf,
                    t,
                    np.absolute(train),
                    np.absolute(recon_xyz),
                    np.absolute(recon_xanes),
                    np.absolute(pred_xyz),
                    np.absolute(pred_xanes),
                )
            )

        print("*" * (16 + 18 * 5))

        fmt = "{:<10s}{:>5.1f}s{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}{:>18.8f}"
        means_ = (
            np.mean(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("mean", *means_))
        stdevs_ = (
            np.std(np.absolute(scores[score]))
            for score in (
                "fit_time",
                "train_score",
                "test_recon_xyz_score",
                "test_recon_xanes_score",
                "test_pred_xyz_score",
                "test_pred_xanes_score",
            )
        )
        print(fmt.format("std. dev.", *stdevs_))
        print("*" * (16 + 18 * 5))

        print("")

    return 0
