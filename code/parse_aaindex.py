""" Parsing the AAIndex database to use as input features in the neural nets """

from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_raw_data():
    """ load and parse the raw aa index data """
    # read the aa index file
    aaindex_fn = "source_data/aaindex/aaindex1"
    with open(aaindex_fn) as f:
        lines = f.readlines()

    # set up an empty dataframe (will append to it)
    data = pd.DataFrame([], columns=["accession number", "description", "A", "C", "D", "E", "F", "G", "H", "I", "K",
                                     "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])

    # the order of amino acids in the aaindex file
    line_1_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I"]
    line_2_order = ["L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    all_entries = []
    current_entry = {}
    reading_aa_props = 0
    for line in lines:
        if line.startswith("//"):
            all_entries.append(current_entry)
            current_entry = {}
        elif line.startswith("H"):
            current_entry.update({"accession number": line.split()[1]})
        elif line.startswith("D"):
            current_entry.update({"description": " ".join(line.split()[1:])})
        elif line.startswith("I"):
            reading_aa_props = 1
        elif reading_aa_props == 1:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_1_order, line.split())})
            reading_aa_props = 2
        elif reading_aa_props == 2:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_2_order, line.split())})
            reading_aa_props = 0

    data = data.append(all_entries)
    return data


def my_pca(pcs, n_components):
    np.random.seed(7)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(pcs)
    print("Captured variance:", sum(pca.explained_variance_ratio_))
    return principal_components


def gen_pca_from_raw_data():

    n_components = 19
    out_fn = "data/aaindex/pca-{}.npy".format(n_components)
    if isfile(out_fn):
        raise FileExistsError("aaindex pca already exists: {}".format(out_fn))

    data = parse_raw_data()

    aas = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    # standardize each aa feature onto unit scale
    aa_features = data.loc[:, aas].values.astype(np.float32)
    # for standardization and PCA, we need it in [n_samples, n_features] format
    aa_features = aa_features.transpose()
    # standardize
    aa_features = StandardScaler().fit_transform(aa_features)

    # pca
    pcs = my_pca(aa_features, n_components=n_components)
    np.save(out_fn, pcs)


def main():
    gen_pca_from_raw_data()


if __name__ == "__main__":
    main()
