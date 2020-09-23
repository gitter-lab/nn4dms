""" parsing arguments to regression.py """

import argparse


def get_parser():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # defining the dataset and network to train
    parser.add_argument("--dataset_name",
                        help="the name of the dataset. this is used for looking up various attributes in constants.py."
                             " i highly recommend you add your dataset to constants.py. if this is arg is not "
                             " specified, you must specify dataset_file, wt_aa, and wt_ofs. this argument takes"
                             " priority over all the other args.",
                        type=str,
                        default="")

    parser.add_argument("--dataset_file",
                        help="if the dataset_name is not given, then this argument is required. the path to the tsv"
                             " dataset file containing the variants and scores",
                        type=str,
                        default="")

    parser.add_argument("--wt_aa",
                        help="if dataset_name is not given, then this argument is required. the full wild-type "
                             "amino acid sequence, used for encoding variants",
                        type=str,
                        default="")

    parser.add_argument("--wt_ofs",
                        help="if dataset_name is not given, then this argument is required. the wild-type offset."
                             " see the data encoding notebook for how this is used.",
                        type=str,
                        default="")

    parser.add_argument("--net_file",
                        help="yaml file containing network spec",
                        type=str,
                        default="network_specs/net_lr.yml")

    parser.add_argument("--encoding",
                        help="which data encoding to use",
                        type=str,
                        default="one_hot,aa_index")

    parser.add_argument("--graph_fn",
                        help="path to the graph file if using a GCN",
                        type=str,
                        default="")

    # training hyperparameters
    parser.add_argument("--learning_rate",
                        help="learning rate",
                        type=float,
                        default=0.0001)

    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=32)

    parser.add_argument("--epochs",
                        help="maximum number of training epochs",
                        type=int,
                        default=300)

    parser.add_argument("--early_stopping",
                        help="set this flag to enable early stopping",
                        action="store_true")

    parser.add_argument("--early_stopping_allowance",
                        help="number of epochs allowance for early stopping",
                        type=int,
                        default=10)

    parser.add_argument("--min_loss_decrease",
                        help="the min amount by which the loss must decrease. if the loss does not decrease by this "
                             "amount for the given allowance of epochs, then training is considered complete",
                        type=float,
                        default=0.00001)

    # defining the train-test-split
    parser.add_argument("--split_dir",
                        help="directory containing the train/tune/test split",
                        type=str,
                        default="")

    parser.add_argument("--train_size",
                        help="size of the training set. used to create a standard train-tune-test split if "
                             "a split_dir is not specified",
                        type=float,
                        default=0.6)

    parser.add_argument("--tune_size",
                        help="size of the tuning set. used to create a standard train-tune-test split if "
                             "a split_dir is not specified",
                        type=float,
                        default=0.2)

    parser.add_argument("--test_size",
                        help="size of the testing set. used to create a standard train-tune-test split if "
                             "a split_dir is not specified",
                        type=float,
                        default=0.2)

    # random seeds
    parser.add_argument("--split_rseed",
                        help="random seed for creating the train-tune-test split. only used if you are creating the"
                             "split on the fly. doesn't apply if you are loading a split from a split_dir.",
                        type=int,
                        default=7)

    parser.add_argument("--py_rseed",
                        help="random seed for python, set just before training begins at the top of run_training()",
                        type=int,
                        default=7)

    parser.add_argument("--np_rseed",
                        help="random seed for numpy, set just before training begins at the top of run_training()",
                        type=int,
                        default=7)

    parser.add_argument("--tf_rseed",
                        help="random seed for tensorflow, set just before training begins at the top of run_training()",
                        type=int,
                        default=7)

    # miscellaneous training, printing, and cleanup arguments
    parser.add_argument("--step_display_interval",
                        help="display step interval",
                        type=float,
                        default=0.1)

    parser.add_argument("--epoch_checkpoint_interval",
                        help="save a model checkpoint every x epochs. if you're using early stopping, make sure this is"
                             "set to 1, otherwise the early stopping checkpoint might not be saved. if not using "
                             "early stopping, it will only keep the latest 2 checkpoints, saved based on this interval",
                        type=int,
                        default=1)

    parser.add_argument("--epoch_evaluation_interval",
                        help="perform evaluation on all splits every x epochs",
                        type=int,
                        default=5)

    parser.add_argument("--delete_checkpoints",
                        help="set this flag to delete checkpoints",
                        action="store_true")

    parser.add_argument("--compress_everything",
                        help="set flag to compress all output except args.txt and final_evaluation.txt",
                        action="store_true")

    parser.add_argument("--log_dir_base",
                        help="log directory base",
                        type=str,
                        default="output/training_logs")

    parser.add_argument("--cluster",
                        help="cluster (when running on HTCondor)",
                        type=str,
                        default="local")

    parser.add_argument("--process",
                        help="process (when running on HTCondor)",
                        type=str,
                        default="local")

    return parser


def save_args(args_dict, out_fn):
    """ save argparse arguments dictionary back to a file that can be used as input to regression.py """
    with open(out_fn, "w") as f:
        for k, v in args_dict.items():
            # ignore these special arguments
            if k not in ["cluster", "process"]:
                # if a flag is set to false, dont include it in the argument file
                if (not isinstance(v, bool)) or (isinstance(v, bool) and v):
                    f.write("--{}\n".format(k))
                    # if a flag is true, no need to specify the "true" value
                    if not isinstance(v, bool):
                        f.write("{}\n".format(v))


def main():
    pass


if __name__ == "__main__":
    main()
