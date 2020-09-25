""" Encodes data in different formats """

from os.path import join, isfile
import argparse

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import constants
import utils


def enc_aa_index(int_seqs):
    """ encodes data in aa index properties format """
    aa_features = np.load("data/aaindex/pca-19.npy")
    # add all zero features for stop codon
    aa_features = np.insert(aa_features, 0, np.zeros(aa_features.shape[1]), axis=0)
    aa_features_enc = aa_features[int_seqs]
    return aa_features_enc


def enc_one_hot(int_seqs):
    enc = OneHotEncoder(categories=[range(constants.NUM_CHARS)] * int_seqs.shape[1], dtype=np.bool, sparse=False)
    one_hot = enc.fit_transform(int_seqs).reshape((int_seqs.shape[0], int_seqs.shape[1], constants.NUM_CHARS))
    return one_hot


def enc_int_seqs_from_char_seqs(char_seqs):
    seq_ints = []
    for char_seq in char_seqs:
        int_seq = [constants.C2I_MAPPING[c] for c in char_seq]
        seq_ints.append(int_seq)
    seq_ints = np.array(seq_ints)
    return seq_ints


def enc_int_seqs_from_variants(variants, wild_type_seq, wt_offset=0):
    # convert wild type seq to integer encoding
    wild_type_int = np.zeros(len(wild_type_seq), dtype=np.uint8)
    for i, c in enumerate(wild_type_seq):
        wild_type_int[i] = constants.C2I_MAPPING[c]

    seq_ints = np.tile(wild_type_int, (len(variants), 1))

    for i, variant in enumerate(variants):
        # special handling if we want to encode the wild-type seq
        # the seq_ints array is already filled with WT, so all we have to do is just ignore it
        # and it will be properly encoded
        if variant == "_wt":
            continue

        # variants are a list of mutations [mutation1, mutation2, ....]
        variant = variant.split(",")
        for mutation in variant:
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])
            replacement = constants.C2I_MAPPING[mutation[-1]]
            seq_ints[i, position-wt_offset] = replacement

    return seq_ints


def encode_int_seqs(char_seqs=None, variants=None, wild_type_aa=None, wild_type_offset=None):
    single = False
    if variants is not None:
        if not isinstance(variants, list):
            single = True
            variants = [variants]

        int_seqs = enc_int_seqs_from_variants(variants, wild_type_aa, wild_type_offset)

    elif char_seqs is not None:
        if not isinstance(char_seqs, list):
            single = True
            char_seqs = [char_seqs]

        int_seqs = enc_int_seqs_from_char_seqs(char_seqs)

    return int_seqs, single


def encode(encoding, char_seqs=None, variants=None, ds_name=None, wt_aa=None, wt_offset=None):
    """ the main encoding function that will encode the given sequences or variants and return the encoded data """

    if variants is None and char_seqs is None:
        raise ValueError("must provide either variants or full sequences to encode")
    if variants is not None and ((ds_name is None) and ((wt_aa is None) or (wt_offset is None))):
        raise ValueError("if providing variants, must also provide (wt_aa and wt_offset) or "
                         "ds_name so I can look up the WT sequence myself")

    if ds_name is not None:
        wt_aa = constants.DATASETS[ds_name]["wt_aa"]
        wt_offset = constants.DATASETS[ds_name]["wt_ofs"]

    # convert given variants or char sequences to integer sequences
    # this may be a bit slower, but easier to program
    int_seqs, single = encode_int_seqs(char_seqs=char_seqs, variants=variants,
                                       wild_type_aa=wt_aa, wild_type_offset=wt_offset)

    # encode variants using int seqs
    encodings = encoding.split(",")
    encoded_data = []
    for enc in encodings:
        if enc == "one_hot":
            encoded_data.append(enc_one_hot(int_seqs))
        elif enc == "aa_index":
            encoded_data.append(enc_aa_index(int_seqs))
        else:
            raise ValueError("err: encountered unknown encoding: {}".format(enc))

    # concatenate if we had more than one encoding
    if len(encoded_data) > 1:
        encoded_data = np.concatenate(encoded_data, axis=-1)
    else:
        encoded_data = encoded_data[0]

    # if we were passed in a single sequence, remove the extra dimension
    if single:
        encoded_data = encoded_data[0]

    return encoded_data


def encode_full_dataset(ds_name, encoding):
    # load the dataset
    ds = utils.load_dataset(ds_name=ds_name)
    # encode the data
    encoded_data = encode(encoding=encoding, variants=ds["variant"].tolist(), ds_name=ds_name)
    return encoded_data


def encode_full_dataset_and_save(ds_name, encoding):
    """ encoding a full dataset """
    out_fn = join(constants.DATASETS[ds_name]["ds_dir"], "enc_{}_{}.npy".format(ds_name, encoding))
    if isfile(out_fn):
        print("err: encoded data already exists: {}".format(out_fn))
        return
    encoded_data = encode_full_dataset(ds_name, encoding)
    np.save(out_fn, encoded_data)
    return encoded_data


def main(args):

    if args.ds_name == "all":
        ds_names = constants.DATASETS.keys()
    else:
        ds_names = [args.ds_name]

    if args.encoding == "all":
        encodings = ["one_hot", "aa_index"]
    else:
        encodings = [args.encoding]

    for ds_name in ds_names:
        for encoding in encodings:
            encode_full_dataset_and_save(ds_name, encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name",
                        help="name of the dataset",
                        type=str)
    parser.add_argument("encoding",
                        help="what encoding to use",
                        type=str,
                        choices=["one_hot", "aa_index", "all"])
    main(parser.parse_args())
