""" Split dataset into train, tune, and test sets """
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
    """ split data into train, tune, and test sets """
    if not np.isclose(train_size + tune_size + test_size, 1):
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


def position_split(ds, seq_len, wt_ofs, train_pos_size, tune_size, rseed=8, out_dir=None, overwrite=False):
    """
    split based on sequence positions
    :param ds: dataset dataframe
    :param seq_len: number of sequence positions
    :param wt_ofs: offset for mutation position, in case dataset variants are not 0-indexed
    :param train_pos_size: fraction of sequence positions to use as training (1-train_pos_size will be used as test)
    :param tune_size: fraction of training samples to use for tuning set (the total number of training samples depends
                    on how many non-overlapping variants are in the selected sequence positions for training)
    :param rseed: random seed
    :param out_dir: output directory where to place the resulting split directory
    :param overwrite: whether to overwrite output if it already exists
    :return: split: the train-tune-test split
             out_dir_split: output directory if the split was saved to disk
             pool_seq_positions: the split of sequence positions into train and test pools
             pool_dataset_idxs: idxs of the variants that fall in each pool (train and test) plus the overlaps (overlap)
    """

    # set the random seed
    np.random.seed(rseed)

    if train_pos_size >= 1 or train_pos_size <= 0:
        raise ValueError("train_pos_size must be in range (0, 1)")

    if tune_size > 1 or tune_size < 0:
        raise ValueError("tune_size must be in range [0, 1] ")

    # determine the number of sequence positions that will be train-set only
    num_train_positions = int(np.round(seq_len * train_pos_size))
    num_test_positions = int(seq_len - num_train_positions)
    logger.info("num_train_positions: {}, num_test_positions: {}".format(num_train_positions, num_test_positions))
    if num_train_positions == 0 or num_test_positions == 0:
        raise RuntimeError("num_train_positions and num_test_positions can't be zero")

    # determine which sequence positions will be train and which will be test
    position_idxs = np.arange(0, seq_len)
    train_positions, test_positions = train_test_split(position_idxs, train_size=num_train_positions)

    # find training, test, and overlapping pools of variants
    train_pool_idxs = []
    test_pool_idxs = []
    overlap_pool_idxs = []
    for idx, variant in enumerate(ds["variant"]):
        muts = variant.split(",")
        mut_positions = [int(mut[1:-1]) - wt_ofs for mut in muts]
        if all(mut_pos in train_positions for mut_pos in mut_positions):
            # train variant
            train_pool_idxs.append(idx)
        elif all(mut_pos in test_positions for mut_pos in mut_positions):
            # test variant
            test_pool_idxs.append(idx)
        else:
            overlap_pool_idxs.append(idx)
    # convert to numpy arrays (for consistency with loading function)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)
    overlap_pool_idxs = np.array(overlap_pool_idxs, dtype=int)
    logger.info("train pool size: {}, test pool size: {}, overlap pool size: {}".format(
        len(train_pool_idxs), len(test_pool_idxs), len(overlap_pool_idxs)))

    # create the actual split
    split = {}
    # split training pool into train and tune sets
    if tune_size > 0:
        if tune_size == 1:
            split["tune"] = train_pool_idxs
        else:
            train_idxs, tune_idxs = train_test_split(train_pool_idxs, test_size=tune_size)
            split["train"] = train_idxs
            split["tune"] = tune_idxs
    else:
        split["train"] = train_pool_idxs
    # the full test pool becomes the test set
    # note: the test set idxs will be sorted because they are coming directly from the test_pool_idxs
    split["test"] = test_pool_idxs

    num_train = 0
    num_tune = 0
    num_test = len(split["test"])
    if "train" in split:
        num_train = len(split["train"])
    if "tune" in split:
        num_tune = len(split["tune"])
    logger.info("num_train: {}, num_tune: {}, num_test: {}".format(num_train, num_tune, num_test))

    # create dictionaries of pool sequence positions and pool dataset idxs to save and return
    pool_seq_positions = {"train": train_positions, "test": test_positions}
    pool_dataset_idxs = {"train": train_pool_idxs, "test": test_pool_idxs, "overlap": overlap_pool_idxs}

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        out_dir_split = join(out_dir, "position_tr-pos{}_tu{}_r{}".format(train_pos_size, tune_size, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-tune-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)

            # save additional info such as the pool sequence positions and the pool dataset idxs
            save_split(pool_seq_positions, join(out_dir_split, "additional_info", "pool_seq_positions"))
            save_split(pool_dataset_idxs, join(out_dir_split, "additional_info", "pool_dataset_idxs"))

    additional_info = {"pool_seq_positions": pool_seq_positions, "pool_dataset_idxs": pool_dataset_idxs}

    return split, out_dir_split, additional_info


def mutation_split(ds, train_muts_size, tune_size, rseed=8, out_dir=None, overwrite=False):

    # set the random seed
    np.random.seed(rseed)

    # all individual mutations in the dataset
    all_mutations = set()
    for variant in ds["variant"]:
        muts = variant.split(",")
        for mut in muts:
            all_mutations.add(mut)
    all_mutations = list(all_mutations)
    num_unique_mutations = len(all_mutations)
    logger.info("number of unique mutations in ds: {}".format(num_unique_mutations))

    # determine the number of mutations positions that will be train-set only
    num_train_mutations = int(np.round(num_unique_mutations * train_muts_size))
    num_test_mutations = int(num_unique_mutations - num_train_mutations)
    logger.info("num_train_mutations: {}, num_test_mutations: {}".format(num_train_mutations, num_test_mutations))

    # sample correct number of train and test mutations
    train_mutations, test_mutations = train_test_split(all_mutations, train_size=num_train_mutations)

    # find training, test, and overlapping pools of variants
    train_pool_idxs = []
    test_pool_idxs = []
    overlap_pool_idxs = []
    for idx, variant in enumerate(ds["variant"]):
        muts = variant.split(",")
        if all(mut in train_mutations for mut in muts):
            train_pool_idxs.append(idx)
        elif all(mut in test_mutations for mut in muts):
            test_pool_idxs.append(idx)
        else:
            overlap_pool_idxs.append(idx)

    # convert to numpy arrays (for consistency with loading function)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)
    overlap_pool_idxs = np.array(overlap_pool_idxs, dtype=int)
    logger.info("train pool size: {}, test pool size: {}, overlap pool size: {}".format(
        len(train_pool_idxs), len(test_pool_idxs), len(overlap_pool_idxs)))

    # create the actual split
    split = {}
    # split training pool into train and tune sets
    if tune_size > 0:
        if tune_size == 1:
            split["tune"] = train_pool_idxs
        else:
            train_idxs, tune_idxs = train_test_split(train_pool_idxs, test_size=tune_size)
            split["train"] = train_idxs
            split["tune"] = tune_idxs
    else:
        split["train"] = train_pool_idxs
    # the full test pool becomes the test set
    split["test"] = test_pool_idxs

    num_train = 0
    num_tune = 0
    num_test = len(split["test"])
    if "train" in split:
        num_train = len(split["train"])
    if "tune" in split:
        num_tune = len(split["tune"])
    logger.info("num_train: {}, num_tune: {}, num_test: {}".format(num_train, num_tune, num_test))

    # create dictionaries of pool mutations & dataset idxs
    pool_muts = {"train": train_mutations, "test": test_mutations}
    pool_dataset_idxs = {"train": train_pool_idxs, "test": test_pool_idxs, "overlap": overlap_pool_idxs}

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        out_dir_split = join(out_dir, "mutation_tr-muts{}_tu{}_r{}".format(train_muts_size, tune_size, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-tune-test split to directory {}".format(out_dir_split))
            # save the main split
            save_split(split, out_dir_split)
            # save additional info
            save_split(pool_muts, join(out_dir_split, "additional_info", "pool_muts"))
            save_split(pool_dataset_idxs, join(out_dir_split, "additional_info", "pool_dataset_idxs"))

    additional_info = {"pool_muts": pool_muts, "pool_dataset_idxs": pool_dataset_idxs}

    return split, out_dir_split, additional_info


def save_split(split, d):
    """ save a split to a directory """
    utils.mkdir(d)
    for k, v in split.items():
        out_fn = join(d, "{}.txt".format(k))
        utils.save_lines(out_fn, v)


def load_single_split_dir(split_dir, content="idxs"):
    # add special exception for hidden files as I was having some problems on some servers (remnant of untarring?)
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if f != "additional_info" and not f.startswith(".")]
    split = {}
    for f in fns:
        # logger.info("loading split from: {}".format(f))
        split_name = basename(f)[:-4]
        if content == "idxs":
            split_data = np.loadtxt(f, delimiter="\n", dtype=int)
        elif content == "txt":
            split_data = utils.load_lines(f)
        else:
            raise ValueError("unsupported content type for split")
        split[split_name] = split_data
    return split


def load_split_dir(split_dir):
    """ load saved splits. has an exception for a "additional_info" directory, but otherwise assumes the given directory
        contains only text files (for a regular train-test-split) or only directories (containing replicates for a
        reduced train size split). any split dirs created with this script should be fine. """

    if not isdir(split_dir):
        raise FileNotFoundError("split directory doesn't exist: {}".format(split_dir))

    # get all the files in the given split_dir
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if f != "additional_info"]

    # reduced train size split dir with multiple split replicates
    if isdir(fns[0]):
        splits = []
        for fn in fns:
            splits.append(load_single_split_dir(fn))
        return splits

    else:
        split = load_single_split_dir(split_dir)
        return split


def load_additional_info(split_dir):
    additional_info_dir = join(split_dir, "additional_info")
    if not isdir(additional_info_dir):
        raise FileNotFoundError("additional_info directory doesn't exist: {}".format(additional_info_dir))

    fns = [join(additional_info_dir, f) for f in os.listdir(additional_info_dir)]
    if len(fns) == 0:
        raise FileNotFoundError("additional_info directory is empty: {}".format(additional_info_dir))

    additional_info = {}
    for fn in fns:
        # key based on specific additional info, leaving option for future additional info in different format
        if basename(fn) in ["pool_dataset_idxs", "pool_seq_positions"]:
            additional_info[basename(fn)] = load_single_split_dir(fn)
        elif basename(fn) in ["pool_muts"]:
            additional_info[basename(fn)] = load_single_split_dir(fn, content="txt")

    return additional_info


def main():
    pass


if __name__ == "__main__":
    main()
