""" general utility functions used throughput codebase """

import os
from os.path import join, isfile, isdir, basename

import numpy as np
import pandas as pd

import constants
import encode

__version__ = '0.2.0'

def mkdir(d):
    """ creates given dir if it does not already exist """
    if not isdir(d):
        os.makedirs(d)


def load_lines(fn):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines


def save_lines(out_fn, lines):
    """ saves each line in data to given file """
    with open(out_fn, "w") as f_handle:
        for line in lines:
            f_handle.write("{}\n".format(line))


def load_dataset(ds_name=None, ds_fn=None):
    """ load a dataset as pandas dataframe """
    if ds_name is None and ds_fn is None:
        raise ValueError("must provide either ds_name or ds_fn to load a dataset")

    if ds_fn is None:
        ds_fn = constants.DATASETS[ds_name]["ds_fn"]

    if not isfile(ds_fn):
        raise FileNotFoundError("can't load dataset, file doesn't exist: {}".format(ds_fn))

    ds = pd.read_csv(ds_fn, sep="\t")
    return ds


def load_encoded_data(ds_name, encoding, gen=False):
    """ load encoded data... specify multiple encodings by making encoding a comma-separated list """
    encodings = encoding.split(",")
    encoded_data = []
    for enc in encodings:
        encd_fn = join(constants.DATASETS[ds_name]["ds_dir"], "enc_{}_{}.npy".format(ds_name, enc))
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
            raise FileNotFoundError("couldn't find {} encoded data, "
                                    "run encode.py to generate: {}".format(enc, encd_fn))

    # concatenate if we had more than one encoding
    if len(encoded_data) > 1:
        encoded_data = np.concatenate(encoded_data, axis=-1)
    else:
        encoded_data = encoded_data[0]

    return encoded_data


def check_equal(iterator):
    """ returns true if all items in iterator are equal, otherwise returns false """

    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def batch_generator(data_arrays, batch_size, skip_last_batch=True, num_epochs=-1, shuffle=True):
    """ generates batches from given data and labels """

    epoch = 0

    if len(data_arrays) == 0:
        raise Exception("No data arrays.")

    data_lens = [len(data_array) for data_array in data_arrays]
    if not check_equal(data_lens):
        raise Exception("Data arrays are different lengths.")

    data_len = data_lens[0]

    batching_data_arrays = data_arrays

    while True:
        # shuffle the input data for this batch
        if shuffle:
            idxs = np.arange(0, data_len)
            np.random.shuffle(idxs)

            batching_data_arrays = []
            for data_array in data_arrays:
                batching_data_arrays.append(data_array[idxs])

        for batch_idx in range(0, data_len, batch_size):

            # skip last batch if it is the wrong size
            if skip_last_batch:
                if batch_idx + batch_size > data_len:
                    continue

            data_batches = []
            for batching_data_array in batching_data_arrays:
                data_batches.append(batching_data_array[batch_idx:(batch_idx + batch_size)])

            yield data_batches

        epoch += 1
        if epoch == num_epochs:
            break


def parse_log_dir_name(log_dir):
    """ simple function for parsing log dirs, if you need to parse the log dir name anywhere you should use this.
        because if you ever update the log dir format in regression.py, you just need to change this once instead of
        going through the whole codebase and searching for where you parse log dir names manually """

    # assuming no surprise underscores from net_file, etc
    tokens = basename(log_dir).split("_")
    parsed = {"date": tokens[3],
              "time": tokens[4],
              "cluster": tokens[1],
              "process": tokens[2],
              "dataset": tokens[5],
              "net": tokens[6],
              "learning_rate": tokens[7],
              "batch_size": tokens[8],
              "uuid": tokens[9]}
    return parsed
