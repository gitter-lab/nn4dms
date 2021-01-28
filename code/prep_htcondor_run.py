""" This is a helper script to set up a run on HTCondor.
    Even if you don't use HTCondor, this might help you set up a run on your queuing system. """

import subprocess
import os
import shutil
from os.path import join, isdir, isfile
import time
import gen_args
import utils


def copy_submit_file(out_dir, num_jobs):
    # prep the sub file by filling in the number of jobs
    with open("htcondor/templates/htcondor.sub", "r") as f:
        data = f.read()
        data = data.replace("NUM_JOBS", str(num_jobs))
    with open(os.path.join(out_dir, "htcondor.sub"), "w") as f:
        f.write(data)


def prep_run(run_name, args_template_file=None, args_dir=None, expand_split_dirs=False):
    run_dir = join("htcondor/htcondor_runs", "run_{}_{}".format(time.strftime("%Y-%m-%d_%H-%M-%S"), run_name))

    if args_template_file is None and args_dir is None:
        raise ValueError("must provide either an args template file or an existing directory containing args files")

    args_out_dir = join(run_dir, "args")
    utils.mkdir(args_out_dir)

    if args_template_file is not None:
        # generate args files
        gen_args.gen_args_from_template_fn(args_template_file, args_out_dir, expand_split_dirs=expand_split_dirs)
        num_args = len(os.listdir(args_out_dir))
    elif args_dir is not None:
        # copy over arguments, renaming to 0....N
        args_fns = [join(args_dir, x) for x in os.listdir(args_dir)]
        num_args = len(args_fns)
        for i, args_fn in enumerate(args_fns):
            filename, extension = os.path.splitext(args_fn)
            shutil.copyfile(args_fn, join(args_out_dir, "{}{}".format(i, extension)))
    # tar and delete the args from the htcondor run dir
    subprocess.call(["tar", "-czf", join(run_dir, "args.tar.gz"), "-C", run_dir, "args"])
    shutil.rmtree(args_out_dir)

    # tar all the current code, data, net specs, anything else needed for the run
    subprocess.call(["tar", "-czf", join(run_dir, "nn4dms.tar.gz"), "code", "data", "network_specs", "pub/splits",
                     "pub/trained_models/gb1"])

    # copy over template files
    shutil.copy("htcondor/templates/run.sh", run_dir)
    shutil.copy("htcondor/templates/env.yml", run_dir)
    # shutil.copy("htcondor/templates/Miniconda3-py37_4.8.2-Linux-x86_64.sh", run_dir)
    copy_submit_file(run_dir, num_args)

    # create empty directories for htcondor logs and training logs (training logs might not be necessary, do it anyway)
    os.makedirs(join(run_dir, "output", "htcondor_logs"))
    os.makedirs(join(run_dir, "output", "training_logs"))


def main():
    # args_template_file = "regression_args/example_run.yml"

    # run_name = "pub_models"
    # args_dir = "pub/regression_args"

    # run_name = "cnn_nofc"
    # args_dir = "regression_args/run_cnn_nofc"
    # prep_run(run_name, args_dir=args_dir)

    run_name = "gb1_pab1_tl"
    args_template_file = "regression_args/gb1_pab1_tl_run.yml"
    prep_run(run_name, args_template_file=args_template_file, expand_split_dirs=True)


if __name__ == "__main__":
    main()
