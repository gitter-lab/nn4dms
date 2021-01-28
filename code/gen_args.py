""" for generating arguments files for regression.py
    can be used for reduced splits, hyperparameter sweeps, HTCondor runs, etc """

import os
from os.path import join, basename, isfile, isdir
import itertools
import argparse

import yaml

from parse_reg_args import save_args
import utils


def parse_yml(yml_fn, expand_split_dirs=False):
    """ parses a yml file, which may contain lists for certain arguments, to potentially
        multiple dictionaries with no lists """

    with open(yml_fn, "r") as f:
        yml = yaml.safe_load(f)

        # expand the reduced split dirs if requested, so that each split dir will have its own job
        # this is basically just so I don't have to manually specify all the split dirs for a big run in the
        # run file. although manually doing it would probably be best since I don't plan on using this feature
        # in the future... oh well.
        if expand_split_dirs:
            yml["split_dir"] = expand_reduced_split_dirs(yml["split_dir"])

        # get the possible options for each arg, and make sure it's a list, even if it's just one possible option
        options = [val if isinstance(val, list) else [val] for val in yml.values()]
        combinations = list(itertools.product(*options))

        # create the dictionaries
        dicts = []
        for combo in combinations:
            d = dict(zip(yml.keys(), combo))
            dicts.append(d)

    # perform replacements of key tokens
    for args_dict in dicts:
        for k, v in args_dict.items():
            # this can be generalized for any token, but we are only using dataset_name for now
            if isinstance(v, str) and "#dataset_name#" in v:
                args_dict[k] = v.replace("#dataset_name#", args_dict["dataset_name"])

    return dicts


def expand_reduced_split_dirs(reduced_split_dirs):
    all_split_dirs = []
    for rsd in reduced_split_dirs:
        fns = [join(rsd, x) for x in os.listdir(rsd)]
        # is this a directory containing split dirs, or actual splits? if split dirs, then append each to the new list
        if isdir(fns[0]):
            for split_dir in fns:
                all_split_dirs.append(split_dir)
        else:
            all_split_dirs.append(rsd)
    return all_split_dirs


def gen_args_from_template_fn(template_fn, out_dir, expand_split_dirs=False):
    utils.mkdir(out_dir)
    args_dicts = parse_yml(template_fn, expand_split_dirs=expand_split_dirs)
    for i, d in enumerate(args_dicts):
        save_args(d, join(out_dir, "{}.txt".format(i)))


def main(args):

    if not isfile(args.f):
        raise FileNotFoundError("couldn't find template file: {}".format(args.f))

    out_dir = args.out_dir if args.out_dir != "" else join("regression_args", basename(args.f).split(".")[0])
    gen_args_from_template_fn(template_fn=args.f, out_dir=out_dir, expand_split_dirs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # defining the dataset and network to train
    parser.add_argument("--f",
                        help="the template yaml file containing the argument options",
                        type=str,
                        default="regression_args/example_run.yml")

    # defining the dataset and network to train
    parser.add_argument("--out_dir",
                        help="the directory where the individual argument files will be placed. if empty, "
                             "the argument files will go in regression_args/<template file name dir>",
                        type=str,
                        default="")

    parsed_args = parser.parse_args()
    main(parsed_args)