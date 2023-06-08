""" this is the run script that executes on the server """

import argparse
import subprocess
import shutil

import numpy as np
import pandas as pd

from gen_rosetta_args import gen_rosetta_args
from parse_energies_txt import parse_multiple
import time


def parse_score(score_sc_fn, output_base):
    """ parses the score.sc file to get overall energy for variant """

    df = pd.read_csv(score_sc_fn, delim_whitespace=True, skiprows=1)

    total_scores = df["total_score"].values.astype(np.float32)
    avg_score = np.mean(total_scores)
    np.save("{}variant_score.npy".format(output_base), avg_score)


def main(args):

    # load the variants that will be processed on this server
    # the variants file contains variants in the format <numerical id> <variant>
    # for example:
    # 3375 Q1K,T24P
    # 3376 Q1K,T24Q
    # 3377 Q1K,T24R
    # 3378 Q1K,T24S
    # 3379 Q1K,T24V
    # 3380 Q1K,T24W
    with open(args.variants_fn, "r") as f:
        ids_variants = f.readlines()

    # wild-type offset is needed since the pdb files are labeled from 1-num_residues, while the
    # datasets I have might be labeled from their start in a larger protein. hard-coded, for now
    if "pab1" in args.pdb_fn:
        wt_offset = 126
    else:
        wt_offset = 0

    # keep track of how long it takes to process vsariant
    run_times = []

    # loop through each variant, model it with rosetta, save results
    for id_variant in ids_variants:

        start = time.time()

        vid, variant = id_variant.split()

        print("Running variant {}: {}".format(vid, variant))

        # copy over PDB file into rosetta working dir and rename it to structure.pdb
        shutil.copyfile(args.pdb_fn, "./rosetta_working_dir/structure.pdb")

        # generate the rosetta arguments for this variant
        gen_rosetta_args(variant, wt_offset, "./rosetta_working_dir")

        # run rosetta via relax.sh - make sure to block until complete
        process = subprocess.Popen("~/code/relax.sh", shell=True)
        process.wait()

        # parse the rosetta energy.txt into npy files and place in output directory
        parse_multiple("./rosetta_working_dir/energy.txt", "./output/{}_".format(vid))

        # parse the score.sc and place into output dir
        parse_score("./rosetta_working_dir/score.sc", "./output/{}_".format(vid))

        # if the flag is set, also copy over the raw score.sc and energy.txt files
        if args.save_raw:
            shutil.copyfile("./rosetta_working_dir/energy.txt", "./output/{}_energy.txt".format(vid))
            shutil.copyfile("./rosetta_working_dir/score.sc", "./output/{}_score.sc".format(vid))

        # clean up the rosetta working dir in preparation for next variant
        process = subprocess.Popen("~/code/clean_up_working_dir.sh", shell=True)
        process.wait()

        run_time = time.time()-start
        run_times.append(run_time)
        print("Processing variant {} took {}".format(vid, run_time))

    # create a final runtimes file for this run
    with open("./output/{}.runtimes".format(args.job_id), "w") as f:
        f.write("Avg runtime per variant: {:.3f}\n".format(np.average(run_times)))
        f.write("Std. dev.: {:.3f}\n".format(np.std(run_times)))
        for run_time in run_times:
            f.write("{:.3f}\n".format(run_time))

    # zip all the outputs and delete
    subprocess.call("tar -czf {}_output.tar.gz *".format(args.job_id), cwd="./output", shell=True)
    subprocess.call("find . ! -name '{}_output.tar.gz' -type f -exec rm -f {} +".format(args.job_id, "{}"), cwd="./output", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("variants_fn",
                        help="the file containing the variants",
                        type=str)

    parser.add_argument("--job_id",
                        help="job id is used to save a diagnostic file",
                        type=str,
                        default="no_job_id")

    parser.add_argument("--pdb_fn",
                        help="path to pdb file",
                        type=str,
                        default="pdb_files/ube4b_clean_0002.pdb")

    parser.add_argument("--save_raw",
                        help="set this to save the raw score.sc and energy.txt files in addition to the parsed ones",
                        action="store_true")

    main(parser.parse_args())
