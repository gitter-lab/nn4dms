""" Some functions to make analysis of trained models easier """

import os
from os.path import join, isfile, isdir, basename

import numpy as np
import pandas as pd

from parse_reg_args import get_parser
import split_dataset as sd
import utils
import constants


def load_predictions(log_dirs, col_names=None, ds=None):
    """ loads the predictions for a trained model from the training log directory.
        returns a dataframe with all the variants, scores, set names, and predictions.
        if passing in a list of log directories, this function expects they are all for the same dataset
        and are using the same train/tune/test set split. this function uses the first log directory in the list
         as the template from which it loads the dataset and train/tune/test split."""

    # if we're passed in a single log directory as a string, convert it to a list for consistent handling
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    # if no column names were specified, automatically generate based on log_dirs indexing
    if col_names is None:
        # using a simple naming scheme of "prediction_index", but maybe I should use the UUID of the model instead?
        col_names = ["prediction"] if len(log_dirs) == 1 else ["prediction_{}".format(i) for i in range(len(log_dirs))]
    # otherwise ensure the column names passed in are a list with the same length as log_dirs
    elif isinstance(col_names, str):
        col_names = [col_names]
    elif len(log_dirs) != len(col_names):
        raise ValueError("the number of log dirs {} should equal the number of column names {}".format(
            len(log_dirs), len(col_names)))

    for i, log_dir in enumerate(log_dirs):
        if not isdir(log_dir):
            raise FileNotFoundError("couldn't find log directory {}".format(log_dir))

        args_fn = join(log_dir, "args.txt")
        split_dir = join(log_dir, "split")
        pred_dir = join(log_dir, "predictions")

        # error checking -- no need to check split_dir since error checking for that is done inside of the
        # sd.load_split_dir() function
        if not isfile(args_fn):
            raise FileNotFoundError("couldn't find args.txt inside of log directory {}".format(args_fn))
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
            ds_fn = constants.DATASETS[args.dataset_name]["ds_fn"] if args.dataset_name != "" else args.dataset_file
            ds = utils.load_dataset(ds_fn=ds_fn)

        # load the split used to train the model and add a column to the dataframe with the set_name
        split = sd.load_split_dir(split_dir)
        if "set_name" not in ds.columns:
            ds["set_name"] = np.nan
            for set_name, idxs in split.items():
                ds.iloc[idxs, ds.columns.get_loc("set_name")] = set_name
        else:
            # check to see if the train/tune/test split is the same
            for set_name, idxs in split.items():
                if not np.all(ds.iloc[idxs, ds.columns.get_loc("set_name")] == set_name):
                    raise ValueError("detected different train/tune/test splits among given log dirs")

        # now go through each set of predictions and add to the appropriate place in the dataframe
        ds[col_names[i]] = np.nan
        for pfn in predictions_fns:
            set_name = basename(pfn).split("_")[0]
            ds.iloc[split[set_name], ds.columns.get_loc(col_names[i])] = np.loadtxt(pfn, delimiter="\n")

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
    log_dir_parsed = utils.parse_log_dir_name(log_dir)
    args["uuid"] = log_dir_parsed["uuid"]
    args["log_dir"] = log_dir

    # add split information to help with processing reduced training size runs -- decided to do this
    # during analysis instead of here in this function
    # split_dir_tokens = args["split_dir"].split("_")
    # split_type = split_dir_tokens[0]
    # args["split_type"] = split_type
    # args["reduced_rep_num"] = int(split_dir_tokens[-1]) if split_type == "reduced" else pd.NA
    # args["reduced_train_prop"] = float(split_dir_tokens[1][2:]) if split_type == "reduced" else pd.NA

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


def load_metrics_and_args(log_dirs):

    # if we're passed in a single log directory as a string, convert it to a list for consistent handling
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    rows = []
    for log_dir in log_dirs:
        m = load_metrics(log_dir)
        args = load_args(log_dir)

        # collapse metrics down to a single row
        ms = m.drop(["epoch", "early"], axis=1).set_index("set").stack().to_frame().T
        ms.columns = ["_".join(c) for c in ms.columns.values]
        ms["epoch"] = m.loc[0, "epoch"]
        ms["early"] = m.loc[0, "early"]
        ms = ms.reindex(columns=["epoch", "early"] + [c for c in ms.columns if c not in ["epoch", "early"]])

        # concat the args and metrics (horizontally)
        row = pd.concat((args.reset_index(drop=True), ms.reset_index(drop=True)), axis=1, ignore_index=False)
        rows.append(row)

    if len(rows) == 1:
        return rows[0]
    else:
        return pd.concat(rows, axis=0).reset_index(drop=True)


def main():
    pass


if __name__ == "__main__":
    main()
