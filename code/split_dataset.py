""" Split dataset into train, validate, and test sets """

import time
from os.path import join, isfile, isdir
from collections import Counter
from collections.abc import Iterable
import hashlib

import numpy as np
import sklearn.model_selection

import utils
import constants


def supertest(ds, size=.1, rseed=8, out_dir=None):
    """ create a supertest split... meant to be completely held out data until final evaluation """
    np.random.seed(rseed)
    idxs = np.arange(0, ds.shape[0])
    idxs, super_test_idxs = sklearn.model_selection.train_test_split(idxs, test_size=size)
    if out_dir is not None:
        utils.mkdir(out_dir)
        out_fn = "supertest_s{}_r{}.txt".format(size, rseed)
        if isfile(join(out_dir, out_fn)):
            raise FileExistsError("supertest split already exists: {}".format(join(out_dir, out_fn)))
        else:
            print("saving supertest split to file {}".format(join(out_dir, out_fn)))
            utils.save_lines(join(out_dir, out_fn), super_test_idxs)
    return np.array(super_test_idxs, dtype=int)


def load_withhold(withhold):
    """ load indices to withhold from split (for supertest set, for example) """
    if isinstance(withhold, str):
        if not isfile(withhold):
            raise FileNotFoundError("err: couldn't find file w/ indices to withhold: {}".format(withhold))
        else:
            withhold = np.loadtxt(withhold, delimiter="\n", dtype=int)
    elif not isinstance(withhold, Iterable):
        raise ValueError("err: withhold must be a string specifying a filename containing indices to withhold"
                         "or an iterable containing those indices")
    return np.array(withhold, dtype=int)


def hash_withhold(withheld_idxs, length=6):
    """ hash the withheld indices for file & directory naming purposes """
    hash_object = hashlib.shake_256(withheld_idxs)
    w = hash_object.hexdigest(length)
    return w


def train_tune_test(ds, train_size=.90, tune_size=.1, test_size=0., withhold=None, rseed=8, out_dir=None):
    """ split data into train, validate, and test sets """
    if train_size + tune_size + test_size != 1:
        raise ValueError("err: train_size, tune_size, and test_size must add up to 1")

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
        split["withheld"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    # validation set
    if tune_size > 0:
        idxs, tune_idxs = sklearn.model_selection.train_test_split(idxs, test_size=tune_size)
        split["tune"] = tune_idxs
    if test_size > 0:
        idxs, test_idxs = sklearn.model_selection.train_test_split(idxs, test_size=test_size)
        split["test"] = test_idxs
    if train_size > 0:
        idxs, train_idxs = sklearn.model_selection.train_test_split(idxs, test_size=train_size)
        split["train"] = train_idxs

    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split["withheld"])
        out_dir_split = join(out_dir, "tr{}_tu{}_te{}_w{}_r{}".format(train_size, tune_size, test_size, w, rseed))
        if isdir(out_dir_split):
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            utils.mkdir(out_dir_split)
            print("saving train-tune-test split to directory {}".format(out_dir_split))
            for k, v in split.items():
                out_fn = join(out_dir_split, "{}.txt".format(k))
                utils.save_lines(out_fn, v)
    return split


def reduced_train_size(ds, tune_size=.1, test_size=0., train_prop=.5, num_train_reps=5,
                       withhold=None, rseed=8, out_dir=None):
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
        split_template["withheld"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    # validation set
    if tune_size > 0:
        idxs, tune_idxs = sklearn.model_selection.train_test_split(idxs, test_size=tune_size)
        split_template["tune"] = tune_idxs
    if test_size > 0:
        idxs, test_idxs = sklearn.model_selection.train_test_split(idxs, test_size=test_size)
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

        train_idxs, unused_idxs = sklearn.model_selection.train_test_split(idxs, train_size=train_prop)
        split_i["train"] = train_idxs
        splits.append(split_i)

    # save out to directory
    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split_template["withheld"])
        out_dir_split = join(out_dir, "reduced_tr{}_tu{}_te{}_w{}_s{}_r{}".format(train_prop, tune_size, test_size, w,
                                                                                  num_train_reps, rseed))
        if isdir(out_dir_split):
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            print("saving reduced split to directory {}".format(out_dir_split))
            for i, split in enumerate(splits):
                out_dir_split_rep = join(out_dir_split, "rep_{}".format(i))
                utils.mkdir(out_dir_split_rep)
                for k, v in split.items():
                    out_fn = join(out_dir_split_rep, "{}.txt".format(k))
                    utils.save_lines(out_fn, v)

    return splits


def main():
    ds = utils.load_dataset("gb1")
    # supertest_idxs = supertest(ds, size=.1, rseed=8, out_dir="data/gb1/splits")
    # split = train_tune_test(ds, train_size=.6, tune_size=.2, test_size=.2,
    #                         withhold="data/gb1/splits/supertest_s0.1_r8.txt",
    #                         rseed=8, out_dir="data/gb1/splits")

    splits = reduced_train_size(ds, tune_size=.1, test_size=.1, train_prop=.00025, num_train_reps=5,
                                withhold="data/gb1/splits/supertest_s0.1_r8.txt", rseed=8, out_dir="data/gb1/splits")

if __name__ == "__main__":
    main()
