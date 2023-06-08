""" Parses Rosetta's output energy.txt """
import argparse

import numpy as np
import pandas as pd


def energies_for_single_pose_id(df_pose_id):
    """ parses the energies per-residue and pairwise energies for a single pose_id """

    # per-residue energies
    per_residue_df = df_pose_id[df_pose_id["restype2"] == "onebody"]
    # columns 8-25 contain the energies
    per_residue_energies = per_residue_df.iloc[:, 8:25].values.astype(np.float32)

    pairwise_df = df_pose_id[df_pose_id["restype2"] != "onebody"]
    # create a matrix where each row is [resi1, resi2, energies...]
    pairwise_energies = pd.concat([pairwise_df[["resi1", "resi2"]], pairwise_df.iloc[:, 8:25]], axis=1).values.astype(np.float32)

    return per_residue_energies, pairwise_energies


def parse_multiple(input_fn, output_base):

    # load the energy.txt file as a pandas dataframe
    df = pd.read_csv(input_fn, delim_whitespace=True)

    # get a list of unique pose_id
    pose_ids = pd.unique(df["pose_id"])

    # loop through each pose id, getting energies for those IDs
    all_per_residue_energies = []
    all_pairwise_energies = []
    for pose_id in pose_ids:
        df_pose_id = df[df["pose_id"] == pose_id]
        per_residue_energies, pairwise_energies = energies_for_single_pose_id(df_pose_id)
        all_per_residue_energies.append(per_residue_energies)
        all_pairwise_energies.append(pairwise_energies)

    # take average of all per residue energies across the different pose ids

    # are any of these different?
    matches = True
    for i in range(len(all_per_residue_energies)-1):
        if not np.all(all_per_residue_energies[i] == all_per_residue_energies[i+1]):
            matches = False
    if matches:
        print("All the per residue energies are exactly the same between the different pose_ids...")

    all_per_residue_energies = np.stack(all_per_residue_energies)
    avg_per_residue_energies = np.mean(all_per_residue_energies, axis=0)
    total_std_dev = np.sum(np.std(all_per_residue_energies, axis=0))
    print("Per-residue standard deviation between pose_ids: {}".format(total_std_dev))
    np.save("{}per_residue_energies.npy".format(output_base), avg_per_residue_energies)

    # for pairwise energies, check if all the different pose_ids have same pairs
    matches = True
    for i in range(len(all_pairwise_energies)-1):
        if not np.all(all_pairwise_energies[i][:, 0:2] == all_pairwise_energies[i+1][:, 0:2]):
            print("Pairwise energies for pose_id {} do not match pose_id {}".format(i, i+1))
            matches = False

    if matches:
        all_pairwise_energies = np.stack(all_pairwise_energies)
        avg_pairwise_energies = np.mean(all_pairwise_energies, axis=0)
        np.save("{}pairwise_energies.npy".format(output_base), avg_pairwise_energies)
    else:
        np.save("{}pairwise_energies.npy".format(output_base), all_pairwise_energies[0])


def main(args):

    parse_multiple(args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input",
                        help="input path and filename",
                        type=str)

    parser.add_argument("--output",
                        help="output path and base name. 'per_residue.npy' and 'pairwise.npy' will be appended",
                        type=str)

    main(parser.parse_args())
