""" Some functions to make analysis of trained models easier """

import os
from os.path import join, isfile, isdir, basename

import numpy as np
import pandas as pd

from parse_reg_args import get_parser
import split_dataset as sd
import utils
import constants


def load_predictions(log_dir, pcol="prediction", ds=None):
    """ loads the predictions for a trained model from the training log directory.
        returns a dataframe with all the variants, scores, set names, and predictions """

    args_fn = join(log_dir, "args.txt")
    split_dir = join(log_dir, "split")
    pred_dir = join(log_dir, "predictions")

    # error checking
    if not isfile(args_fn):
        raise FileNotFoundError("couldn't find args.txt inside of log directory {}".format(args_fn))
    if not isdir(split_dir):
        raise FileNotFoundError("couldn't find the split dir inside the log dir {}".format(split_dir))
    if not isdir(pred_dir):
        raise FileNotFoundError("couldn't find the predictions dir inside of the log dir {}".format(pred_dir))

    # the predictions dir is also going to have true scores
    # we don't need those because we are loading them directly from the dataset tsv
    predictions_fns = [join(pred_dir, x) for x in os.listdir(pred_dir) if x.split("_")[1] == "predicted"]
    if len(predictions_fns) == 0:
        raise FileNotFoundError("couldn't find any predicted scores in the predictions dir {}".format(pred_dir))

    # load the arguments used to train this model
    parser = get_parser()
    args = parser.parse_args(["@{}".format(args_fn)])

    # load the full dataset used to train the model
    if ds is None:
        dataset_file = constants.DATASETS[args.dataset_name]["ds_fn"] if args.dataset_name != "" else args.dataset_file
        if not isfile(dataset_file):
            raise FileNotFoundError("couldn't find dataset file {}".format(dataset_file))
        ds = utils.load_dataset(ds_fn=dataset_file)

    # load the split used to train the model and add a column to the dataframe with the set_name
    ds["set_name"] = np.nan
    split = sd.load_split_dir(split_dir)
    for set_name, idxs in split.items():
        ds.iloc[idxs, ds.columns.get_loc("set_name")] = set_name

    # now go through each set of predictions and add to the appropriate place in the dataframe
    ds[pcol] = np.nan
    for pfn in predictions_fns:
        set_name = basename(pfn).split("_")[0]
        ds.iloc[split[set_name], ds.columns.get_loc(pcol)] = np.loadtxt(pfn, delimiter="\n")

    return ds


def load_args(log_dir):
    """ loads the args used to train a model in the form of a dataframe.
        this is not meant to be used to re-train models, rather it's for analysis purposes.
        the args may contain extra info that arent passed in to regression.py, like the UUID """
    args_fn = join(log_dir, "args.txt")
    if not isfile(args_fn):
        raise FileNotFoundError("couldn't find args.txt inside of log directory {}".format(args_fn))

    # load the arguments used to train this model
    parser = get_parser()
    args = vars(parser.parse_args(["@{}".format(args_fn)]))  # also convert to a dict

    # add the UUID and log_dir to the args
    # todo: rather than relying on this split to get the uuid, there should be a utils function that parses the log dir,
    #  which can be updated if the log dir format changes for whatever reason
    uuid = basename(log_dir).split("_")[-1]
    args["uuid"] = uuid
    args["log_dir"] = log_dir

    args_df = pd.DataFrame(args, index=[0])
    # move uuid and other index-like items to first column
    c2m = ["uuid", "cluster", "process", "log_dir"]
    args_df = args_df.reindex(columns=c2m + [c for c in args_df.columns if c not in c2m])

    return args_df


def load_metrics(log_dir):
    """ loads the pre-computed metrics (inside final_evaluation.txt) along with the arguments
        used to train the model """

    eval_fn = join(log_dir, "final_evaluation.txt")
    if not isfile(eval_fn):
        raise FileNotFoundError("couldn't find the final evaluation: {}".format(eval_fn))

    metrics = pd.read_csv(eval_fn, delimiter="\t")
    return metrics


def load_metrics_and_args(log_dir):
    m = load_metrics(log_dir)
    args = load_args(log_dir)

    # collapse metrics down to a single row
    ms = m.drop(["epoch", "early"], axis=1).set_index("set").stack().to_frame().T
    ms.columns = ["_".join(c) for c in ms.columns.values]
    ms["epoch"] = m.loc[0, "epoch"]
    ms["early"] = m.loc[0, "early"]
    ms = ms.reindex(columns=["epoch", "early"] + [c for c in ms.columns if c not in ["epoch", "early"]])

    # concat the args and metrics
    final = pd.concat((args, ms), axis=1, ignore_index=False)
    return final




def main():
    log_dir = "output/training_logs/log_local_local_d-avgfp_n-lr_lr-0.0001_bs-128_2020-09-09_11-23-13"
    ds = load_predictions(log_dir)
    print(ds)


if __name__ == "__main__":
    main()
