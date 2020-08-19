""" general utility functions used throughput codebase """

import os
from os.path import join, isfile

import numpy as np
import pandas as pd

import constants
import encode


def ensure_dir_exists(d):
    """ creates given dir if it does not already exist """
    if not os.path.isdir(d):
        os.makedirs(d)


def load_lines(fn):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines


def load_dataset(ds_name=None, ds_fn=None):
    """ load a dataset as pandas dataframe """
    if ds_name is None and ds_fn is None:
        raise ValueError("err: must provide either ds_name or ds_fn to load a dataset")

    if ds_fn is None:
        ds_fn = constants.DS_FNS[ds_name]

    ds = pd.read_csv(ds_fn, sep="\t")
    return ds


def load_encoded_data(ds_name, encoding, gen=False):
    """ load encoded data... specify multiple encodings by making encoding a comma-separated list """
    encodings = encoding.split(",")
    encoded_data = []
    for enc in encodings:
        encd_fn = join(constants.DS_DIRS[ds_name], "enc_{}_{}.npy".format(ds_name, enc))
        if isfile(encd_fn):
            encd = np.load(encd_fn)
            encoded_data.append(encd)
        elif not isfile(encd_fn) and gen:
            print("err: couldn't find {} encoded data, "
                  "run encode.py to generate: {}".format(enc, encd_fn))
            print("generating encoded data for one-time use...")
            encd = encode.encode_full_dataset(ds_name, enc)
            encoded_data.append(encd)
        else:
            raise FileNotFoundError("err: couldn't find {} encoded data, "
                                    "run encode.py to generate: {}".format(enc, encd_fn))

    # concatenate if we had more than one encoding
    if len(encoded_data) > 1:
        encoded_data = np.concatenate(encoded_data, axis=-1)
    else:
        encoded_data = encoded_data[0]

    return encoded_data
