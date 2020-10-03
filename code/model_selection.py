import os
from os.path import join, basename, isdir
from glob import glob
import logging
import shutil

import pandas as pd
import numpy as np

import analysis as an
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nn4dms." + __name__)
logger.setLevel(logging.DEBUG)


def selection(data, constants, criteria, want_min=True):
    """ perform model selection and return """

    grouped = data.groupby(constants)

    if want_min:
        selected_idxs = grouped[criteria].idxmin()
    else:
        selected_idxs = grouped[criteria].idxmax()

    print(selected_idxs)
    selected_data = data.loc[selected_idxs]
    return selected_idxs, selected_data


def copy_to_selected_dir(container_log_dirs, selected_data, output_dir):

    # build up an index linking uuid --> directory so that it is trivial to find the directory for given uuid
    ident_dir_map = {}
    # load the model log dirs, determine uuids, and make dictionary mapping
    # todo: might be able to make this more general by using glob -- what if there's only a single container log dir?
    # what if the container log dirs don't point exactly to the training_logs dir, and we just want to find all
    # directories starting with "log" and ending with a UUID? -- like if the they just point to run dirs instead?
    model_log_dirs = []
    for cld in container_log_dirs:
        for mld in os.listdir(cld):
            model_log_dirs.append(join(cld, mld))
    # build the mapping
    for mld in model_log_dirs:
        uuid = utils.parse_log_dir_name(mld)["uuid"]
        ident_dir_map[uuid] = mld

    # move any log dirs that have not yet been moved to the output dir
    for uuid in selected_data["uuid"]:
        source_mld = ident_dir_map[uuid]
        out_mld = join(output_dir, basename(source_mld))
        if isdir(out_mld):
            logger.info("log directory has already previously copied over, skipping: {}".format(out_mld))
        else:
            logger.info("moving {} to the selected output directory:".format(basename(source_mld)))
            shutil.copytree(source_mld, out_mld)


def cnn_nofc_selection():
    """ perform selection for the cnn nofc sweep and move trained models to separate dir """

    out_dir = "output/selected_models_cnn_nofc"

    # training log directories for all datasets
    container_log_dirs = ["output/run_2020-10-01_17-22-22_bgl3_cnn_nofc_sweep/output/training_logs",
                          "output/run_2020-10-01_17-21-00_avgfp_cnn_nofc_sweep/output/training_logs",
                          "output/run_2020-10-01_17-22-36_gb1_cnn_nofc_sweep/output/training_logs",
                          "output/run_2020-10-01_17-22-44_pab1_cnn_nofc_sweep/output/training_logs",
                          "output/run_2020-10-01_17-22-53_ube4b_cnn_nofc_sweep/output/training_logs"]

    # load the data from all of these runs into a single dataframe that we can perform selection on in one go
    model_log_dirs = []
    for cld in container_log_dirs:
        for mld in os.listdir(cld):
            model_log_dirs.append(join(cld, mld))
    nofc_data = an.load_metrics_and_args(model_log_dirs)

    # perform the selection
    selected_idxs, selected_data = selection(nofc_data, constants=["dataset_name"], criteria="tune_mse", want_min=True)
    print(selected_data[["uuid", "dataset_name", "batch_size", "learning_rate", "tune_mse", "stest_pearsonr"]])

    copy_to_selected_dir(container_log_dirs, selected_data, out_dir)


def main():
    cnn_nofc_selection()


if __name__ == "__main__":
    main()
