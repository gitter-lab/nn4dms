""" Parsing source data into simple tsv datasets.
    To parse Bgl3 and GB1, ENRICH2 MUST BE INSTALLED IN A SEPARATE CONDA ENVIRONEMNT NAMED 'enrich2' """

from os.path import isfile, join
import collections

import numpy as np
import pandas as pd

import enrich2
import utils


def parse_avgfp():
    """ create the gfp dataset from raw source data """
    source_fn = "source_data/avgfp/amino_acid_genotypes_to_brightness.tsv"
    out_fn = "data/avgfp/avgfp.tsv"
    if isfile(out_fn):
        print("err: parsed avgfp dataset already exists: {}".format(out_fn))
        return

    # load the source data
    data = pd.read_csv(source_fn, sep="\t")

    # remove the wild-type entry
    data = data.loc[1:]

    # create columns for variants, number of mutations, and score
    variants = data["aaMutations"].apply(lambda x: ",".join([x[1:] for x in x.split(":")]))
    num_mutations = variants.apply(lambda x: len(x.split(",")))
    score = data["medianBrightness"]

    # create the dataframe
    cols = ["variant", "num_mutations", "score"]
    data_dict = {"variant": variants.values, "num_mutations": num_mutations.values, "score": score.values}
    df = pd.DataFrame(data_dict, columns=cols)

    # now add a normalized score column - these scores have the wild-type score subtracted from them
    df["score_wt_norm"] = df["score"].apply(lambda x: x - 3.7192121319)

    df.to_csv(out_fn, sep="\t", index=False)


def filter_dataset(df, threshold):
    """ filter out variants that do not meet the required threshold for number of reads """
    df = df[(df["inp"] + df["sel"]) >= threshold]
    return df


def parse_bgl3_variant_list(ml, col_name):
    """ creates a dataframe from the given list of variants """
    # filter wild-type counts out, add to dataframe at the end
    ml_no_wt = []
    wt = []
    for variant in ml:
        if variant.startswith("WTcount"):
            wt.append(int(variant.split(",")[-1].strip()))
        else:
            ml_no_wt.append(variant)

    count_dict = collections.Counter(ml_no_wt)
    frame = pd.DataFrame(index=count_dict.keys(), data=count_dict.values())
    frame.columns = [col_name]

    # add wild-type counts back in to datafrae
    frame.loc["_wt"] = sum(wt)

    return frame


def get_bgl3_count_df(output_dir=None):
    """ combines the inp and sel variant lists into a single dataframe with counts """
    inp_fn = "source_data/bgl3/unlabeled_Bgl3_mutations.txt"
    sel_fn = "source_data/bgl3/positive_Bgl3_mutations.txt"

    cache_fn = "bgl3_raw_counts.tsv"
    if output_dir is None or not isfile(join(output_dir, cache_fn)):
        print("Computing bgl3 count df from raw counts")
        inp_variant_list = utils.load_lines(inp_fn)
        sel_variant_list = utils.load_lines(sel_fn)
        df = pd.concat([parse_bgl3_variant_list(inp_variant_list, "inp"),
                        parse_bgl3_variant_list(sel_variant_list, "sel")], axis=1, sort=True).fillna(0)
        if output_dir is not None:
            df.to_csv(join(output_dir, cache_fn), sep="\t")
        return df

    print("Loading cached count df from file: {}".format(join(output_dir, cache_fn)))
    return pd.read_csv(join(output_dir, cache_fn), sep="\t", index_col=0)


def parse_bgl3():
    """ create the bgl3 dataset from raw source data """
    out_dir = "data/bgl3"
    out_fn = "bgl3.tsv"
    if isfile(join(out_dir, out_fn)):
        print("err: parsed bgl3 dataset already exists: {}".format(join(out_dir, out_fn)))
        return

    # creates a single dataframe with counts from the given mutations lists
    df = get_bgl3_count_df(output_dir=out_dir)

    # filter the variants based on count threshold
    threshold = 5
    df = filter_dataset(df, threshold=threshold)

    enrich2.create_e2_dataset(df, output_dir=out_dir, output_fn=out_fn)


def get_gb1_count_df(output_dir=None):
    """ creates a single dataframe with raw counts for all gb1 variants """
    cache_fn = "gb1_raw_counts.tsv"
    if output_dir is None or not isfile(join(output_dir, cache_fn)):
        print("Computing gb1 count df from raw counts")

        single = pd.read_csv("source_data/gb1/single_mutants.csv")
        double = pd.read_csv("source_data/gb1/double_mutants.csv")
        wt = pd.read_csv("source_data/gb1/wild_type.csv")

        # change position to a 0-index instead of the current 1-index
        single["Position"] = single["Position"] - 1
        double["Mut1 Position"] = double["Mut1 Position"] - 1
        double["Mut2 Position"] = double["Mut2 Position"] - 1

        single_strings = single.apply(lambda row: "".join(map(str, row[0:3])), axis=1)
        double_strings = double.apply(lambda row: "{}{}{},{}{}{}".format(*row[0:6]), axis=1)
        wt_string = pd.Series(["_wt"])

        combined_strings = pd.concat([single_strings, double_strings, wt_string], axis=0).reset_index(drop=True)
        combined_input_count = pd.concat([single["Input Count"], double["Input Count"], wt["Input Count"]], axis=0).reset_index(drop=True)
        combined_selection_count = pd.concat([single["Selection Count"], double["Selection Count"], wt["Selection Count"]], axis=0).reset_index(drop=True)

        # save a combined all variants file with variant and counts
        cols = ["variant", "inp", "sel"]
        data = {"variant": combined_strings.values, "inp": combined_input_count.values, "sel": combined_selection_count.values}
        df = pd.DataFrame(data, columns=cols)
        df = df.set_index("variant")

        if output_dir is not None:
            df.to_csv(join(output_dir, cache_fn), sep="\t")

        return df

    print("Loading cached count df from file: {}".format(join(output_dir, cache_fn)))
    return pd.read_csv(join(output_dir, cache_fn), sep="\t", index_col=0)


def parse_gb1():
    """ create the gb1 dataset from raw source data """
    out_dir = "data/gb1"
    out_fn = "gb1.tsv"
    if isfile(join(out_dir, out_fn)):
        print("err: parsed gb1 dataset already exists: {}".format(join(out_dir, out_fn)))
        return

    df = get_gb1_count_df(output_dir=out_dir)
    threshold = 5

    df = filter_dataset(df, threshold)
    enrich2.create_e2_dataset(df, output_dir=out_dir, output_fn=out_fn)


def parse_pab1():
    """ create the pab1 dataset from raw source data """

    # Pab1 sequence starts at 126, but for simplicity in encoding and array access we will offset it to zero
    pab1_wt_offset = 126
    single_mutants_fn = "source_data/pab1/single_mutants_linear.csv"
    double_mutants_fn = "source_data/pab1/double_mutants.csv"
    out_fn = "data/pab1/pab1.tsv"
    if isfile(out_fn):
        print("err: parsed pab1 dataset already exists: {}".format(out_fn))
        return

    single = pd.read_csv(single_mutants_fn, skiprows=1)
    single = single.dropna(how='all')  # remove rows where all values are missing
    double = pd.read_csv(double_mutants_fn)

    # NOTE: Using the LINEAR scores here, these are not log ratios

    # build up the wild type sequence when looking through single mutants
    wt_seq = []

    # single mutants and scores
    single_variants = []
    single_scores = []
    aa_order = single.columns.values[2:]
    for row in single.itertuples():
        wt = row.Residue
        wt_seq.append(wt)
        pos = int(row.position)
        for rep, score in zip(aa_order, row[3:]):
            if not pd.isnull(score):
                # print(wt, pos, rep, score)
                single_variants.append("{}{}{}".format(wt, pos, rep))
                single_scores.append(score)

    # double mutants and scores
    double_variants = []
    double_scores = []
    for row in double.itertuples():
        pos1, rep1 = row.seqID_X.split("-")
        wt1 = wt_seq[int(pos1) - pab1_wt_offset]
        pos2, rep2 = row.seqID_Y.split("-")
        wt2 = wt_seq[int(pos2) - pab1_wt_offset]

        variant = "{}{}{},{}{}{}".format(wt1, pos1, rep1, wt2, pos2, rep2)
        double_variants.append(variant)
        score = row.XY_Enrichment_score
        double_scores.append(score)

    # combine single and double
    all_variants = single_variants + double_variants
    all_scores = np.array(single_scores + double_scores)
    num_mutations = [len(x.split(",")) for x in all_variants]

    # we want the log2-transformed data
    all_scores = np.log2(all_scores)

    cols = ["variant", "num_mutations", "score"]
    data_dict = {"variant": all_variants, "num_mutations": num_mutations, "score": all_scores}
    df = pd.DataFrame(data_dict, columns=cols)

    df.to_csv(out_fn, sep="\t", index=False)


def convert_ube4b_seqid(seqid):
    """ convert the ube4b sequence ids in the raw data to the standard mutation format """
    ube4b_wt = "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLPSGTVMDRSIILRHLLNSPTDPFNRQMLTESMLEPVPELKEQIQAWMREKQSSDH"
    positions, replacements = seqid.split("-")
    positions = [int(p) for p in positions.split(",")]
    replacements = replacements.split(",")

    wpr = []
    for pos, rep in zip(positions, replacements):
        wpr.append("{}{}{}".format(ube4b_wt[pos], pos, rep))

    return ",".join(wpr)


def parse_ube4b():
    """ create the ube4b dataset from raw source data """

    raw_data_fn = "source_data/ube4b/ube4b.xlsx"
    out_fn = "data/ube4b/ube4b.tsv"
    if isfile(out_fn):
        print("err: parsed ube4b dataset already exists: {}".format(out_fn))
        return

    data = pd.read_excel(raw_data_fn)
    data = data.drop(data.index[19347])
    data = data.dropna()

    wpr_seq_ids = data["seqID"].map(convert_ube4b_seqid)
    num_mutations = [len(x.split(",")) for x in wpr_seq_ids]

    cols = ["variant", "num_mutations", "score"]
    data_dict = {"variant": wpr_seq_ids, "num_mutations": num_mutations, "score": data["nscor_log2_ratio"]}
    df = pd.DataFrame(data_dict, columns=cols)

    df.to_csv(out_fn, sep="\t", index=False)


def parse(ds_name):
    """ parse given dataset name """
    parse_funs = {"avgfp": parse_avgfp,
                  "bgl3": parse_bgl3,
                  "gb1": parse_gb1,
                  "pab1": parse_pab1,
                  "ube4b": parse_ube4b}

    if ds_name == "all":
        for k, v in parse_funs.items():
            v()
    elif ds_name in parse_funs:
        parse_funs[ds_name]()
    else:
        print("err: invalid ds_name")


def main():
    parse("all")


if __name__ == "__main__":
    main()
