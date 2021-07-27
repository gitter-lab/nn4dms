""" for generating arguments files for regression.py
    can be used for reduced splits, hyperparameter sweeps, HTCondor runs, etc """

import os
from os.path import join, basename, isfile, isdir
import itertools
import argparse
import re
import copy

import yaml

from parse_reg_args import save_args
import utils


def parse_keyargs(yml):
    ymls = []
    for keyargs_key in yml["keyargs"]["keys"]:
        yml_for_key = copy.deepcopy(yml)
        for k, v in yml.items():
            if isinstance(v, str):
                # search for any tokens for replacement in all string args
                rep_tokens = re.findall("(<key:(.*?)>)", v)
                # supporting multiple keyargs in a single string means if there are
                # multiple args defined for any of the keys, then we have to do a combination
                if len(rep_tokens) > 1:
                    raise ValueError("multiple keyargs per string are not supported")
                elif len(rep_tokens) == 1:
                    rep_token, rep_key = rep_tokens[0]
                    # print(rep_token, rep_key)
                    yml_for_key[k] = yml["keyargs"][rep_key][keyargs_key]
        del yml_for_key["keyargs"]
        ymls.append(yml_for_key)

    return ymls


def parse_yml(yml_fn, expand_split_dirs=False):
    """ parses a yml file, which may contain lists for certain arguments, to potentially
        multiple dictionaries with no lists """

    # todo: better way to handle the complex replace functionality, maybe using a template manager
    # it may not function correctly when using keyargs and the old replace system

    with open(yml_fn, "r") as f:
        raw_yml = yaml.safe_load(f)

        # parse keyargs to create multiple ymls
        if "keyargs" in raw_yml:
            ymls = parse_keyargs(raw_yml)
        else:
            ymls = [raw_yml]

        dicts = []
        for yml in ymls:
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
            for combo in combinations:
                d = dict(zip(yml.keys(), combo))
                dicts.append(d)

    # perform replacements of key tokens
    for args_dict in dicts:
        for k, v in args_dict.items():
            if isinstance(v, str):
                # search for any tokens for replacement in all string args
                rep_tokens = re.findall("{.*?}", v)
                for token in rep_tokens:
                    # needs to be a corresponding argument for each token
                    # user can specify normal program arguments or special arguments in the form of {argument}
                    # arguments in the form of {argument} get deleted and are not in the final generated args file
                    if token in args_dict.keys():
                        rep_value = args_dict[token]
                    elif token[1:-1] in args_dict.keys():
                        rep_value = args_dict[token[1:-1]]
                    else:
                        raise ValueError("couldn't find specified token {} in args {}".format(token, args_dict))
                    # perform the replacement
                    args_dict[k] = v.replace(token, rep_value)

    # args in the form of {argument} are just there for token replacements, so they can be removed
    for args_dict in dicts:
        for k, v in args_dict.items():
            if k.startswith("{") and k.endswith("}"):
                del args_dict[k]

    return dicts


def expand_reduced_split_dirs(reduced_split_dirs):
    # depending on how the split_dir was specified in the yaml file, we might be getting a single string or a list
    if isinstance(reduced_split_dirs, str):
        reduced_split_dirs = [reduced_split_dirs]

    all_split_dirs = []
    for rsd in reduced_split_dirs:
        fns = [join(rsd, x) for x in os.listdir(rsd)]
        # is this a directory containing split dirs, or actual splits? if split dirs, then append each to the new list
        # add an exception for "additional_info" dirs in splits that contain it (mutation-based, position-based)
        if isdir(fns[0]) and basename(fns[0]) != "additional_info":
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