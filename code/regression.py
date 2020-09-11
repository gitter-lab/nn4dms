""" fully connected neural network """

# python
import argparse
import collections
import logging
import os
import os.path
from os.path import join, dirname, basename, isdir
import shutil
import sys
import tarfile
import time
import random
import json
from enum import Enum

# 3rd party
import numpy as np
import pandas as pd
import tensorflow as tf

# mine
import utils
import split_dataset as sd
import encode as enc
from build_tf_model import build_graph_from_args_dict
import metrics
import constants

logger = logging.getLogger("nn4dms." + __name__)
logger.setLevel(logging.INFO)


def compute_loss(sess, igraph, tgraph, data, set_name, batch_size):
    """ computes the average loss over all data batches """

    bg = utils.batch_generator((data["encoded_data"][set_name], data["scores"][set_name]), batch_size,
                               skip_last_batch=False, num_epochs=1, shuffle=False)

    loss_vals = []
    for batch_num, batch_data in enumerate(bg):
        ed_batch, scores_batch = batch_data

        # fill the feed dict with the next batch
        feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: ed_batch,
                     igraph["ph_inputs_dict"]["scores"]: scores_batch}

        # get compute the loss for this batch
        loss_val = sess.run(tgraph["loss"], feed_dict=feed_dict)
        loss_vals.append(loss_val)

    # return the average loss across each batch
    return np.average(loss_vals)


def run_eval(sess, args, igraph, tgraph, data, set_name):
    """ runs one evaluation against the full epoch of data """

    # get variables to make transition to new igraph, tgraph system easier
    metrics_ph_dict = tgraph["metrics_ph_dict"]
    summaries_metrics = tgraph["summaries_metrics"]

    bg = utils.batch_generator((data["encoded_data"][set_name], data["scores"][set_name]),
                               args.batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)

    # get all the predicted and true labels in batches
    predicted_scores = np.zeros(data["scores"][set_name].shape)
    true_scores = np.zeros(data["scores"][set_name].shape)

    start = time.time()
    for batch_num, batch_data in enumerate(bg):
        ed_batch, sc_batch = batch_data

        # fill the feed dict with the next batch
        feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: ed_batch,
                     igraph["ph_inputs_dict"]["scores"]: sc_batch}

        # start and end index for this batch
        start_index = batch_num * args.batch_size
        end_index = start_index + args.batch_size

        # get predicted labels for evaluating metrics using sklearn
        preds = sess.run(igraph["predictions"], feed_dict=feed_dict)
        predicted_scores[start_index:end_index] = preds
        true_scores[start_index:end_index] = sc_batch
    duration = time.time() - start

    evaluation_dict = metrics.compute_metrics(true_scores, predicted_scores)

    # update summaries by running metrics ops
    if summaries_metrics is not None:
        metrics_feed_dict = {metrics_ph_dict["mse"]: evaluation_dict["mse"],
                             metrics_ph_dict["pearsonr"]: evaluation_dict["pearsonr"],
                             metrics_ph_dict["r2"]: evaluation_dict["r2"],
                             metrics_ph_dict["spearmanr"]: evaluation_dict["spearmanr"]}
        out = sess.run([summaries_metrics,
                        metrics_ph_dict["mse"],
                        metrics_ph_dict["pearsonr"],
                        metrics_ph_dict["r2"],
                        metrics_ph_dict["spearmanr"]],
                       feed_dict=metrics_feed_dict)
        summary_str = out[0]
    else:
        summary_str = None

    # add summary string to evaluation dict
    evaluation_dict["summary"] = summary_str

    # print
    print("  Evaluation completed in {:.3} sec.".format(duration))
    print("  MSE: {:.3f}".format(evaluation_dict["mse"]))
    print("  pearsonr: {:.3f}".format(evaluation_dict["pearsonr"]))
    print("  r2: {:.3f}".format(evaluation_dict["r2"]))
    print("  spearmanr: {:.3f}".format(evaluation_dict["spearmanr"]))

    return evaluation_dict


def evaluate(sess, args, igraph, tgraph, epoch, data, set_names, summary_writers):
    """ perform evaluation on the given sets, printing output & saving to summary writer """

    # dictionary to store results of evaluations
    evaluations = {}

    for set_name in set_names:
        print("Evaluation: {}".format(set_name))
        evaluations[set_name] = run_eval(sess, args, igraph, tgraph, data, set_name)

        if epoch is not None and (set_name == "train" or set_name == "tune"):
            summary_writers[set_name].add_summary(evaluations[set_name]["summary"], epoch)
            summary_writers[set_name].flush()

    return evaluations


def get_step_display_interval(args, num_train_examples):
    # compute the absolute step display interval
    num_batches = num_train_examples // args.batch_size
    step_display_interval = int(num_batches * args.step_display_interval)
    if step_display_interval == 0:
        step_display_interval = 1
    return step_display_interval


def run_training(data, log_dir, args):

    # reset the current graph and reset all the seeds before training
    tf.compat.v1.reset_default_graph()
    logger.info("setting random seeds py={}, np={}, tf={}".format(args.py_rseed, args.np_rseed, args.tf_rseed))
    random.seed(args.py_rseed)
    np.random.seed(args.np_rseed)
    tf.compat.v1.set_random_seed(args.tf_rseed)

    ed = data["encoded_data"]
    sc = data["scores"]

    # build tensorflow computation graph
    igraph, tgraph = build_graph_from_args_dict(args, encoded_data_shape=ed["train"].shape, reset_graph=False)

    # convert dictionary entries to variables to make transition easier
    init_global = tgraph["init_global"]
    train_op = tgraph["train_op"]
    loss = tgraph["loss"]
    summaries_per_epoch = tgraph["summaries_per_epoch"]
    validation_loss_ph = tgraph["validation_loss_ph"]
    training_loss_ph = tgraph["training_loss_ph"]

    # get the step display interval
    step_display_interval = get_step_display_interval(args, len(ed["train"]))

    # Create a saver for writing training checkpoints.
    max_to_keep = args.early_stopping_allowance + 1 if args.early_stopping else 2
    saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=max_to_keep)

    with tf.compat.v1.Session() as sess:

        # instantiate a summary writers to output summaries for tensorboard
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        summary_writers = {"train": tf.compat.v1.summary.FileWriter(join(log_dir, "train")),
                           "tune": tf.compat.v1.summary.FileWriter(join(log_dir, "validation"))}

        # run the op to initialize the variables
        sess.run(init_global)

        # keep track of when lowest loss... will be used for stopping criteria
        epoch_with_lowest_validate_loss = 1
        lowest_validate_loss = None
        validate_loss_last_epoch = None
        train_loss_last_epoch = None
        num_epochs_since_lowest_validate_loss = 0

        # start the training loop
        logger.info("starting training loop")
        for epoch in range(1, args.epochs + 1):

            # flush stdout at the start of each epoch -- seems to help a bit with htcondor log files?
            sys.stdout.flush()

            # keep track of real time for this epoch
            epoch_start_time = time.time()

            # keep some statistics for this epoch
            epoch_step_durations = []
            epoch_train_loss_values = []

            # keep some statistics for this step interval
            start_step = 1
            interval_step_durations = []
            interval_train_loss_values = []

            # generate the data batches
            bg = utils.batch_generator((ed["train"], sc["train"]),
                                       args.batch_size, skip_last_batch=False, num_epochs=1)

            # loop through each batch of data in this epoch
            for step, batch_data in enumerate(bg):
                ed_batch, sc_batch = batch_data

                step += 1
                step_start_time = time.time()

                # create the feed dictionary to feed batch inputs into the graph
                feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: ed_batch,
                             igraph["ph_inputs_dict"]["scores"]: sc_batch,
                             igraph["ph_inputs_dict"]["training"]: True}

                # run one step of the model
                _, train_loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                # maintain statistics - step duration and loss vals
                step_duration = time.time() - step_start_time
                epoch_step_durations.append(step_duration)
                interval_step_durations.append(step_duration)
                epoch_train_loss_values.append(train_loss_value)
                interval_train_loss_values.append(train_loss_value)

                # display statistics for this step interval and update the events file
                if step % step_display_interval == 0:
                    avg_step_duration = np.average(interval_step_durations)
                    interval_avg_train_loss = np.average(interval_train_loss_values)
                    interval_stat_str = "Epoch {:3} Steps {:4} - {:<4}: Avg Step = {:.2f} Avg TLoss = {:.4f}"
                    print(interval_stat_str.format(epoch, start_step, step, avg_step_duration, interval_avg_train_loss))

                    # reset the interval statistics
                    interval_step_durations = []
                    interval_train_loss_values = []
                    start_step = step + 1

            # end of the epoch - compute loss decrease on training set
            avg_train_loss = compute_loss(sess, igraph, tgraph, data, "train", args.batch_size)
            train_loss_decrease_last_epoch = 0 if train_loss_last_epoch is None else train_loss_last_epoch - avg_train_loss
            train_loss_last_epoch = avg_train_loss

            # end of epoch - compute loss on tune set to check for early stopping
            validate_loss = compute_loss(sess, igraph, tgraph, data, "tune", args.batch_size)
            # the decrease in validation loss since the last epoch
            validate_loss_decrease_last_epoch = 0 if validate_loss_last_epoch is None else validate_loss_last_epoch - validate_loss
            # the decrease in validation loss since the lowest validation loss recorded
            validate_loss_decrease_thresh = 0 if lowest_validate_loss is None else lowest_validate_loss - validate_loss
            # update the validation loss from the last epoch to be the loss at this epoch
            validate_loss_last_epoch = validate_loss

            # stopping criteria - loss doesn't decrease by at least 0.00001 for more than 10 epochs in a row
            if (lowest_validate_loss is None) or ((lowest_validate_loss - validate_loss) > args.min_loss_decrease):
                lowest_validate_loss = validate_loss
                num_epochs_since_lowest_validate_loss = 0
                epoch_with_lowest_validate_loss = epoch
            else:
                num_epochs_since_lowest_validate_loss += 1

            # duration statistics
            epoch_duration = time.time() - epoch_start_time
            avg_step_duration = np.average(epoch_step_durations)

            print("====================")
            print("= Epoch: {:3}".format(epoch))
            print("= Duration: {:.2f}".format(epoch_duration))
            print("= Avg Step Duration: {:.4f}".format(avg_step_duration))
            print("= Training Loss: {:.6f}".format(avg_train_loss))
            print("= Training Loss Decrease (last epoch): {:.6f}".format(train_loss_decrease_last_epoch))
            print("= Validation Loss: {:.6f}".format(validate_loss))
            print("= Validation Loss Decrease (last epoch): {:.6f}".format(validate_loss_decrease_last_epoch))
            print("= Validation Loss Decrease (threshold): {:.6f}".format(validate_loss_decrease_thresh))
            print("= Num Epochs Since Lowest Validation Loss: {}".format(num_epochs_since_lowest_validate_loss))
            print("====================")

            # add per epoch summaries
            summary_str = sess.run(summaries_per_epoch, feed_dict={validation_loss_ph: validate_loss,
                                                                   training_loss_ph: avg_train_loss})
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()

            # save a checkpoint periodically or if it's the last epoch
            if epoch % args.epoch_checkpoint_interval == 0 or epoch == args.epochs:
                save_checkpoint(sess, saver, log_dir, epoch)

            # evaluate the model periodically, or if it's the last epoch
            if epoch % args.epoch_evaluation_interval == 0 or epoch == args.epochs:
                evaluations = evaluate(sess, args, igraph, tgraph, epoch, data, ed.keys(), summary_writers)

                # hit the last epoch, save its evaluation
                if epoch == args.epochs:
                    save_metrics_evaluations(evaluations, log_dir, epoch, early=False, args=args)
                    clean_up_checkpoints(epoch, epoch, log_dir, delete_checkpoints=args.delete_checkpoints,
                                         compress_checkpoints=False)

                    if args.compress_everything:
                        compress_everything(log_dir)

                    return evaluations

            # did we meet the stopping criteria?
            met_early_stopping_allowance = num_epochs_since_lowest_validate_loss == args.early_stopping_allowance

            if args.early_stopping and met_early_stopping_allowance:
                print("V Loss hasn't decreased by more than {} for {} epochs in a row.".format(
                    args.min_loss_decrease, args.early_stopping_allowance))

                print("Training complete.")

                # if we didn't already save a checkpoint for this epoch, save one now
                # this is not the "best" epoch, just the latest for when training ended
                if epoch % args.epoch_checkpoint_interval != 0 and epoch != args.epochs:
                    save_checkpoint(sess, saver, log_dir, epoch)

                # if we didn't already evaluate this epoch, evaluate now (this is just for printing purposes)
                if epoch % args.epoch_evaluation_interval != 0 and epoch != args.epochs:
                    evaluate(sess, args, igraph, tgraph, epoch, data, ed.keys(), summary_writers)

                # load the best model and evaluate it
                print("Loading best model (epoch {}) and evaluating it.".format(epoch_with_lowest_validate_loss))
                load_checkpoint(sess, saver, log_dir, epoch_with_lowest_validate_loss)
                # we pass "None" as the epoch so that the evaluate function doesn't add this evaluation to TensorBoard
                evaluations = evaluate(sess, args, igraph, tgraph, None, data, ed.keys(), summary_writers)

                # save the evaluations
                save_metrics_evaluations(evaluations, log_dir, epoch_with_lowest_validate_loss, early=True, args=args)

                clean_up_checkpoints(epoch, epoch_with_lowest_validate_loss, log_dir,
                                     delete_checkpoints=args.delete_checkpoints, compress_checkpoints=False)
                if args.compress_everything:
                    compress_everything(log_dir)

                return evaluations


def clean_up_checkpoints(epoch, best_epoch, log_dir, delete_checkpoints=True, compress_checkpoints=True):
    """ deletes all checkpoints except the latest and best, compresses them """

    if delete_checkpoints:
        for fn in os.listdir(log_dir):
            if fn.startswith("model.ckpt") or fn == "checkpoint":
                os.remove(join(log_dir, fn))

    else:
        # delete all checkpoints except the latest and best
        for fn in os.listdir(log_dir):
            if fn.startswith("model.ckpt"):
                if "-{}.".format(epoch) not in fn and "-{}.".format(best_epoch) not in fn:
                    os.remove(join(log_dir, fn))

        if compress_checkpoints:
            # tar the latest and best
            with tarfile.open(join(log_dir, "models.tar.gz"), "w:gz") as tar:
                for fn in os.listdir(log_dir):
                    if fn.startswith("model.ckpt"):
                        tar.add(join(log_dir, fn), arcname=fn)

            # delete the latest and best
            for fn in os.listdir(log_dir):
                if fn.startswith("model.ckpt"):
                    os.remove(join(log_dir, fn))


def compress_everything(log_dir):
    """ compresses all output in the log dir except final_evaluation.txt and args.txt """

    exclusion = ["final_evaluation.txt", "args.txt"]

    # tar the latest and best
    with tarfile.open(join(log_dir, "output.tar.gz"), "w:gz") as tar:
        for fn in os.listdir(log_dir):
            if fn not in exclusion:
                tar.add(join(log_dir, fn), arcname=fn)

    exclusion.append("output.tar.gz")

    # delete the latest and best
    for fn in os.listdir(log_dir):
        if fn not in exclusion:
            if os.path.isfile(join(log_dir, fn)):
                os.remove(join(log_dir, fn))
            elif os.path.isdir(join(log_dir, fn)):
                shutil.rmtree(join(log_dir, fn))


def save_scores(evaluation, fn_base):
    np.savetxt("{}_predicted_scores.txt".format(fn_base), evaluation["predicted"], fmt="%f")
    np.savetxt("{}_true_scores.txt".format(fn_base), evaluation["true"], fmt="%f")


def save_metrics_evaluations(evaluations, log_dir, epoch, early, args):
    """ saves metrics for all evaluations """

    # process evaluations to remove the summary, predicted, and true scores
    p_evaluations = {}
    for set_name, evaluation in evaluations.items():
        evaluation = {metric_name: value for metric_name, value in evaluation.items()
                      if metric_name not in ["summary", "predicted", "true"]}
        evaluation["epoch"] = epoch
        evaluation["early"] = early
        p_evaluations[set_name] = evaluation

    # create a pandas dataframe of the evaluation metrics to save as a tsv
    metrics_df = pd.DataFrame(p_evaluations).transpose()
    metrics_df.index.rename("set", inplace=True)
    metrics_df.to_csv(join(log_dir, "final_evaluation.txt"), sep="\t")

    for set_name, evaluation in evaluations.items():
        # save true scores and actual scores predicted using this model
        out_dir = join(log_dir, "predictions")
        utils.mkdir(out_dir)
        save_scores(evaluation, join(out_dir, set_name))


def save_checkpoint(sess, saver, log_dir, epoch):
    checkpoint_file = join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_file, global_step=epoch)


def load_checkpoint(sess, saver, log_dir, epoch):
    checkpoint_file = join(log_dir, "model.ckpt-{}".format(epoch))
    saver.restore(sess, checkpoint_file)


def log_dir_name(args):
    # log directory captures the cluster & process (if running on HTCondor), the dataset name, the
    # network specification file basename, the learning rate, the batch size, and the date and time
    log_dir_str = "log_{}_{}_d-{}_n-{}_lr-{}_bs-{}_{}"

    # just use the arg file basename
    net_arg = basename(args.net_file)[4:-5]

    # dataset file basename if no dataset_name is specified
    if args.dataset_name != "":
        ds_arg = args.dataset_name
    else:
        ds_arg = basename(args.dataset_file)[:-4]

    format_args = [args.cluster, args.process, ds_arg, net_arg,
                   args.learning_rate, args.batch_size, time.strftime("%Y-%m-%d_%H-%M-%S")]

    log_dir = join(args.log_dir_base, log_dir_str.format(*format_args))

    # log directory already exists. so just append a number to it.
    # should only happen if you run the script within the same second with the same args.
    # but handle this edge case just in case.
    if isdir(log_dir):
        log_dir = log_dir + "_2"
    while isdir(log_dir):
        log_dir = "_".join(log_dir.split("_")[:-1] + [str(int(log_dir.split("_")[-1]) + 1)])
        if not isdir(log_dir):
            break
    return log_dir


def save_args(args_dict, log_dir):
    """ save argparse arguments to a file """
    with open(join(log_dir, "args.txt"), "w") as f:
        for k, v in args_dict.items():
            # ignore these special arguments
            if k not in ["cluster", "process"]:
                # if a flag is set to false, dont include it in the argument file
                if (not isinstance(v, bool)) or (isinstance(v, bool) and v):
                    f.write("--{}\n".format(k))
                    # if a flag is true, no need to specify the "true" value
                    if not isinstance(v, bool):
                        f.write("{}\n".format(v))


def main(args):
    """ set up params, log dir, splits, encode the data, and run the training """

    # set up log directory & save the args file to it
    log_dir = log_dir_name(args)
    logger.info("log directory is {}".format(log_dir))
    utils.mkdir(log_dir)
    save_args(vars(args), log_dir)

    # load the dataset
    if args.dataset_name != "":
        dataset_file = constants.DATASETS[args.dataset_name]["ds_fn"]
    else:
        dataset_file = args.dataset_file
    logger.info("loading dataset from {}".format(dataset_file))
    ds = utils.load_dataset(ds_fn=dataset_file)

    # load the dataset split or create one
    if isdir(args.split_dir):
        logger.info("loading split from {}".format(args.split_dir))
        split = sd.load_split_dir(args.split_dir)
        if isinstance(split, list):
            raise ValueError("this script doesn't support multiple reduced train size replicates in a single run. "
                             "run each one individually by specifying the split dir of the replicate. ")
    else:
        # create a classic train-tune-test split based on the specified args
        logger.info("creating a train/test split with tr={}, tu={}, and te={}, seed={}".format(
            args.train_size, args.tune_size, args.test_size, args.split_rseed
        ))
        split = sd.train_tune_test(ds, train_size=args.train_size, tune_size=args.tune_size,
                                   test_size=args.test_size, rseed=args.split_rseed)

    # error checking for split -- make sure we have a train set
    if "train" not in split:
        raise ValueError("no train set in dataset split. specify a split with a train set to proceed.")
    if "tune" not in split:
        raise ValueError("no tune set in dataset split. specify a split with a tune set to proceed. "
                         "the tune set is used for early stopping and logging statistics to tensorboard. "
                         "if you dont want a tune set, and instead just prefer to have a train and test set, "
                         "just name your test set as the tune set so it is compatible with the script. ")

    # save the split indices that are going to be used for this model to the log directory for the model
    # this isn't as good as explicitly saving a split using split_dataset.py because the directory name will
    # not be informative. todo if loading a split_dir, it would be good to copy over the directory name
    logger.info("backing up split to log dir {}".format(join(log_dir, "split")))
    sd.save_split(split, join(log_dir, "split"))

    # figure out the wt_aa and wt_offset for encoding data
    if args.dataset_name != "":
        wt_aa = constants.DATASETS[args.dataset_name]["wt_aa"]
        wt_ofs = constants.DATASETS[args.dataset_name]["wt_ofs"]
    else:
        wt_aa = args.wt_aa
        wt_ofs = args.wt_ofs

    # create the dataset dictionary, containing encoded data, scores, etc, based on the splits
    data = collections.defaultdict(dict)
    data["ds"] = ds
    for set_name, idxs in split.items():
        data["idxs"][set_name] = idxs
        data["variants"][set_name] = ds.iloc[idxs]["variant"].tolist()
        # we are using "score" as the default target, but support for multiple scores could be added here
        data["scores"][set_name] = ds.iloc[idxs]["score"].to_numpy()
        # encode the data
        logger.info("encoding {} set variants using {} encoding".format(set_name, args.encoding))
        data["encoded_data"][set_name] = enc.encode(encoding=args.encoding, variants=data["variants"][set_name],
                                                    wt_aa=wt_aa, wt_offset=wt_ofs)

    evaluations = run_training(data, log_dir, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')

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
                        help="json file containing network spec",
                        type=str,
                        default="network_specs/net_lr.json")

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
                        help="size of the training set",
                        type=float,
                        default=0.6)

    parser.add_argument("--tune_size",
                        help="size of the tuning set",
                        type=float,
                        default=0.2)

    parser.add_argument("--test_size",
                        help="size of the testing set",
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

    parsed_args = parser.parse_args()
    if parsed_args.dataset_name == "":
        if parsed_args.dataset_file == "" or parsed_args.wt_aa == "" or parsed_args.wt_ofs == "":
            parser.error("you must specify either a dataset_name (for a dataset defined in constants.py) or "
                         "all three of the dataset_file, the wt_aa, and the wt_ofs. if you specify the dataset_name,"
                         "it takes priority over all the other args.")
    main(parsed_args)
