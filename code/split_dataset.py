""" Split dataset into train, validate, and test sets """
from os.path import join, isfile, isdir, basename
import os
from collections.abc import Iterable
import hashlib
import logging

import numpy as np
from sklearn.model_selection import train_test_split

import utils


logger = logging.getLogger("nn4dms." + __name__)
logger.setLevel(logging.DEBUG)


def supertest(ds, size=.1, rseed=8, out_dir=None, overwrite=False):
    """ create a supertest split... meant to be completely held out data until final evaluation """
    np.random.seed(rseed)
    idxs = np.arange(0, ds.shape[0])
    idxs, super_test_idxs = train_test_split(idxs, test_size=size)
    save_fn = None
    if out_dir is not None:
        utils.mkdir(out_dir)
        out_fn = "supertest_w{}_s{}_r{}.txt".format(hash_withhold(super_test_idxs), size, rseed)
        save_fn = join(out_dir, out_fn)
        if isfile(save_fn) and not overwrite:
            raise FileExistsError("supertest split already exists: {}".format(join(out_dir, out_fn)))
        else:
            logger.info("saving supertest split to file {}".format(save_fn))
            utils.save_lines(save_fn, super_test_idxs)
    return np.array(super_test_idxs, dtype=int), save_fn


def load_withhold(withhold):
    """ load indices to withhold from split (for supertest set, for example) """
    if isinstance(withhold, str):
        if not isfile(withhold):
            raise FileNotFoundError("couldn't find file w/ indices to withhold: {}".format(withhold))
        else:
            withhold = np.loadtxt(withhold, delimiter="\n", dtype=int)
    elif not isinstance(withhold, Iterable):
        raise ValueError("withhold must be a string specifying a filename containing indices to withhold "
                         "or an iterable containing those indices")
    return np.array(withhold, dtype=int)


def hash_withhold(withheld_idxs, length=6):
    """ hash the withheld indices for file & directory naming purposes """
    hash_object = hashlib.shake_256(withheld_idxs)
    w = hash_object.hexdigest(length)
    return w


def train_tune_test(ds, train_size=.90, tune_size=.1, test_size=0., withhold=None,
                    rseed=8, out_dir=None, overwrite=False):
    """ split data into train, validate, and test sets """
    if train_size + tune_size + test_size != 1:
        raise ValueError("train_size, tune_size, and test_size must add up to 1. current values are "
                         "tr={}, tu={}, and te={}".format(train_size, tune_size, test_size))

    # set the random seed
    np.random.seed(rseed)

    # keep track of all the splits we make
    split = {}

    # set up the indices that will get split
    idxs = np.arange(0, ds.shape[0])

    # withhold supertest data if specified -- can be either a file specifying idxs or an iterable with idxs
    if withhold is not None:
        withhold = load_withhold(withhold)
        # the withheld indices will be saved as part of the split for future reference
        split["stest"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    if tune_size > 0:
        if tune_size == 1:
            split["tune"] = idxs
        else:
            idxs, tune_idxs = train_test_split(idxs, test_size=tune_size)
            split["tune"] = tune_idxs
    if test_size > 0:
        adjusted_test_size = np.around(test_size / (1 - tune_size), 5)
        if adjusted_test_size == 1:
            split["test"] = idxs
        else:
            idxs, test_idxs = train_test_split(idxs, test_size=adjusted_test_size)
            split["test"] = test_idxs
    if train_size > 0:
        adjusted_train_size = np.around(train_size / (1 - tune_size - test_size), 5)
        if adjusted_train_size == 1:
            split["train"] = idxs
        else:
            idxs, train_idxs = train_test_split(idxs, test_size=adjusted_train_size)
            split["train"] = train_idxs

    out_dir_split = None
    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split["stest"])
        out_dir_split = join(out_dir, "standard_tr{}_tu{}_te{}_w{}_r{}".format(train_size, tune_size, test_size, w, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            logger.info("saving train-tune-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)
    return split, out_dir_split


def reduced_train_size(ds, tune_size=.1, test_size=0., train_prop=.5, num_train_reps=5,
                       withhold=None, rseed=8, out_dir=None, overwrite=False):
    """ create splits for reduced train size model. this function first removes any withholding. then it computes
        the tune and test splits using the full amount of data remaining after removing the withholding. then from the
        remaining data, it generates the requested number of train replicates using the requested proportion.
        if specifying the same random seed, should result in the same tune and test splits as train_test_split().
        further, if you specify a train prop of 1, this is equivalent to doing a regular train-tune-test split where
        train_size = 1 - (tune_size + test_size) """

    # set the random seed
    np.random.seed(rseed)

    # each replicate will still have the same tune and test splits
    split_template = {}

    # set up the indices that will get split
    idxs = np.arange(0, ds.shape[0])

    # withhold supertest data if specified -- can be either a file specifying idxs or an iterable with idxs
    if withhold is not None:
        withhold = load_withhold(withhold)
        # the withheld indices will be saved as part of the split for future reference
        split_template["stest"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    if tune_size > 0:
        idxs, tune_idxs = train_test_split(idxs, test_size=tune_size)
        split_template["tune"] = tune_idxs
    if test_size > 0:
        adjusted_test_size = np.around(test_size / (1 - tune_size), 5)
        idxs, test_idxs = train_test_split(idxs, test_size=adjusted_test_size)
        split_template["test"] = test_idxs

    # there will be num_train_reps splits
    splits = []
    for i in range(num_train_reps):
        # make a copy of the template containing tune, test, and withheld indices
        split_i = split_template.copy()
        if train_prop == 1:
            # train proportion of 1 is a special case where all remaining indices not in tune, test, or withheld
            # are placed in the test set. This is equivalent to doing a regular train-tune-test split where
            # train_size = 1 - (tune_size + test_size)
            split_i["train"] = idxs
            # no need to have multiple replicates here since each replicate would have the same training data
            break

        train_idxs, unused_idxs = train_test_split(idxs, train_size=train_prop)
        split_i["train"] = train_idxs
        splits.append(split_i)

    # save out to directory
    out_dir_split = None
    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split_template["stest"])
        out_dir_split = join(out_dir, "reduced_tr{}_tu{}_te{}_w{}_s{}_r{}".format(train_prop, tune_size, test_size, w,
                                                                                  num_train_reps, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            logger.info("saving reduced split to directory {}".format(out_dir_split))
            for i, split in enumerate(splits):
                out_dir_split_rep = join(out_dir_split, basename(out_dir_split) + "_rep_{}".format(i))
                save_split(split, out_dir_split_rep)

    return splits, out_dir_split


def save_split(split, d):
    """ save a split to a directory """
    utils.mkdir(d)
    for k, v in split.items():
        out_fn = join(d, "{}.txt".format(k))
        utils.save_lines(out_fn, v)


def load_single_split_dir(split_dir):
    fns = [join(split_dir, f) for f in os.listdir(split_dir)]
    split = {}
    for f in fns:
        split_name = basename(f)[:-4]
        split_idxs = np.loadtxt(f, delimiter="\n", dtype=int)
        split[split_name] = split_idxs
    return split


def load_split_dir(split_dir):
    """ load saved splits. assumes the given directory contains only text files (for a regular train-test-split)
        or only directories (containing replicates for a reduced train size split). any split dirs created with
        this script should be fine. """

    if not isdir(split_dir):
        raise FileNotFoundError("split directory doesn't exist: {}".format(split_dir))

    # get all the files in the given split_dir
    fns = [join(split_dir, f) for f in os.listdir(split_dir)]

    # reduced train size split dir with multiple split replicates
    if isdir(fns[0]):
        splits = []
        for fn in fns:
            splits.append(load_single_split_dir(fn))
        return splits

    else:
        split = load_single_split_dir(split_dir)
        return split


def main():
    ds = utils.load_dataset("gb1")
    # supertest_idxs = supertest(ds, size=.1, rseed=8, out_dir="data/gb1/splits")
    # split = train_tune_test(ds, train_size=.6, tune_size=.2, test_size=.2,
    #                         withhold="data/gb1/splits/supertest_s0.1_r8.txt",
    #                         rseed=8, out_dir="data/gb1/splits")

    splits, _ = reduced_train_size(ds, tune_size=.1, test_size=.1, train_prop=.00025, num_train_reps=5,
                                withhold="data/avgfp/splits/supertest_s0.1_r8.txt", rseed=15, out_dir="data/avgfp/splits")

if __name__ == "__main__":
    main()
