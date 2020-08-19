""" Split dataset into train, validate, and test sets """

import time
from os.path import join, isfile, isdir
from collections import Counter
from collections.abc import Iterable

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
            print("err: supertest split already exists: {}".format(join(out_dir, out_fn)))
        else:
            print("saving supertest split to file {}".format(join(out_dir, out_fn)))
            utils.save_lines(join(out_dir, out_fn), super_test_idxs)
    return super_test_idxs


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
        if isinstance(withhold, str):
            if not isfile(withhold):
                raise FileNotFoundError("err: couldn't find file w/ indices to withhold: {}".format(withhold))
            else:
                withhold = np.loadtxt(withhold, delimiter="\n", dtype=int)
        elif not isinstance(withhold, Iterable):
            raise ValueError("err: withhold must be a string specifying a filename containing indices to withhold"
                             "or an iterable containing those indices")
        idxs = sorted(set(idxs) - set(withhold))
        split["withheld"] = withhold

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
        out_dir_split = join(out_dir, "tr{}_tu{}_te{}_w{}_r{}".format(train_size, tune_size, test_size,
                                                                      "T" if withhold is not None else "F", rseed))
        if isdir(out_dir_split):
            print("err: split already exists: {}. if you want to withhold a different set of variants, "
                  "i recommend specifying a different out_dir other than {}".format(out_dir_split, out_dir))
        else:
            utils.mkdir(out_dir_split)
            print("saving train-tune-test split to directory {}".format(out_dir_split))
            for k, v in split.items():
                out_fn = join(out_dir_split, "{}.txt".format(k))
                utils.save_lines(out_fn, v)
    return split


def main():
    ds = utils.load_dataset("avgfp")
    supertest_idxs = supertest(ds, size=.1, rseed=8, out_dir="data/avgfp/splits")
    # train_tune_test(ds, train_size=.6, tune_size=.2, test_size=.2, withhold="data/avgfp/splits/supertest_s0.1_r8.txt",
    #                 rseed=8, out_dir="data/avgfp/splits")

    train_tune_test(ds, train_size=.6, tune_size=.2, test_size=.2, withhold=None,
                    rseed=8, out_dir="data/avgfp/splits")


if __name__ == "__main__":
    main()
