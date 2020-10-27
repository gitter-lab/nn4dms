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

# 3rd party
import numpy as np
import pandas as pd
import tensorflow as tf
import uuid
import shortuuid

# mine
import utils
import split_dataset as sd
import encode as enc
from build_tf_model import build_graph_from_args_dict
from parse_reg_args import get_parser, save_args
import metrics
import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nn4dms." + __name__)
logger.setLevel(logging.INFO)


def compute_loss(sess, igraph, tgraph, data, set_name, batch_size):
    """ computes the average loss over all data batches """

    bg = utils.batch_generator((data["encoded_data"][set_name], data["scores"][set_name]), batch_size,
                               skip_last_batch=False, num_epochs=1, shuffle=False)

    loss_vals = []
    num_examples_per_batch = []
    for batch_num, batch_data in enumerate(bg):
        ed_batch, scores_batch = batch_data
        num_examples_per_batch.append(len(scores_batch))

        # fill the feed dict with the next batch
        feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: ed_batch,
                     igraph["ph_inputs_dict"]["scores"]: scores_batch}

        # get compute the loss for this batch
        loss_val = sess.run(tgraph["loss"], feed_dict=feed_dict)
        loss_vals.append(loss_val)

    # return the average loss across each batch
    return np.average(loss_vals, weights=num_examples_per_batch)


def run_eval(sess, args, igraph, tgraph, data, set_name):
    """ runs one evaluation against the full epoch of data """

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

    print("Evaluation ({} set) completed in {:.3} sec.".format(set_name, duration))

    return evaluation_dict


def evaluate(sess, args, igraph, tgraph, epoch, data, set_names, summary_writers):
    """ perform evaluation on the given sets, printing output & saving to summary writer """

    metrics_ph_dict = tgraph["metrics_ph_dict"]
    summaries_metrics = tgraph["summaries_metrics"]

    # dictionary to store results of evaluations
    evaluations = {}

    for set_name in set_names:
        evaluations[set_name] = run_eval(sess, args, igraph, tgraph, data, set_name)

        # update metrics on tensorboard if this is the train or tune set
        # in future, could add all sets (sure, why not), but would need to make summary writers for all
        if epoch is not None and (set_name == "train" or set_name == "tune"):
            metrics_feed_dict = {metrics_ph_dict["mse"]: evaluations[set_name]["mse"],
                                 metrics_ph_dict["pearsonr"]: evaluations[set_name]["pearsonr"],
                                 metrics_ph_dict["r2"]: evaluations[set_name]["r2"],
                                 metrics_ph_dict["spearmanr"]: evaluations[set_name]["spearmanr"]}
            summary_str = sess.run(summaries_metrics, feed_dict=metrics_feed_dict)

            summary_writers[set_name].add_summary(summary_str, epoch)
            summary_writers[set_name].flush()

    # now print out the evaluations as a single dataframe
    print("====================")
    print(evaluations_dict_to_dataframe(evaluations))
    print("====================")

    return evaluations


def get_step_display_interval(args, num_train_examples):
    # compute the absolute step display interval
    num_batches = num_train_examples // args.batch_size
    step_display_interval = int(num_batches * args.step_display_interval)
    if step_display_interval == 0:
        step_display_interval = 1
    return step_display_interval


def run_training_epoch(sess, args, igraph, tgraph, data, epoch, step_display_interval):

    # keep some statistics for this epoch
    epoch_step_durations = []
    epoch_num_examples_per_batch = []

    # keep some statistics for each step interval (reset after each display)
    start_step = 1
    interval_step_durations = []
    interval_train_loss_values = []
    interval_num_examples_per_batch = []

    # generate the data batches
    bg = utils.batch_generator((data["encoded_data"]["train"], data["scores"]["train"]),
                               args.batch_size, skip_last_batch=False, num_epochs=1)

    # loop through each batch of data in this epoch
    for step, batch_data in enumerate(bg):
        ed_batch, sc_batch = batch_data
        epoch_num_examples_per_batch.append(len(sc_batch))
        interval_num_examples_per_batch.append(len(sc_batch))

        step += 1
        step_start_time = time.time()

        # create the feed dictionary to feed batch inputs into the graph
        feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: ed_batch,
                     igraph["ph_inputs_dict"]["scores"]: sc_batch,
                     igraph["ph_inputs_dict"]["training"]: True}

        # run one step of the model
        _, train_loss_value = sess.run([tgraph["train_op"], tgraph["loss"]], feed_dict=feed_dict)

        # maintain statistics - step duration and loss vals
        step_duration = time.time() - step_start_time
        epoch_step_durations.append(step_duration)
        interval_step_durations.append(step_duration)
        interval_train_loss_values.append(train_loss_value)

        # display statistics for this step interval
        if step % step_display_interval == 0:
            avg_step_duration = np.average(interval_step_durations, weights=interval_num_examples_per_batch)
            interval_avg_train_loss = np.average(interval_train_loss_values, weights=interval_num_examples_per_batch)
            interval_stat_str = "Epoch {:3} Steps {:4} - {:<4}: Avg Step = {:.4f} Avg TLoss = {:.4f}"
            print(interval_stat_str.format(epoch, start_step, step, avg_step_duration, interval_avg_train_loss))

            # reset the interval statistics
            interval_step_durations = []
            interval_train_loss_values = []
            start_step = step + 1
            interval_num_examples_per_batch = []

    avg_step_duration = np.average(epoch_step_durations, weights=epoch_num_examples_per_batch)

    return avg_step_duration


class LossTracker:
    # keep track of loss for early stopping and in general... could probably be implemented a bit better
    epochs = []
    train_losses = []
    validate_losses = []

    # keep track of the epoch with the lowest validate loss
    epoch_with_lowest_validate_loss = 1
    lowest_validate_loss = None
    num_epochs_since_lowest_validate_loss = 0

    # the decrease in validation loss since the lowest validation loss recorded
    validate_loss_decrease_thresh = 0

    def __init__(self, mld):
        self.mld = mld

    def add_train_loss(self, epoch, new_train_loss):
        self.epochs.append(epoch)
        self.train_losses.append(new_train_loss)

    def get_train_loss_decrease(self):
        # the decrease in training loss since the last epoch
        if len(self.train_losses) < 2:
            # if no losses have been reported or if only one loss has been reported, there is no loss decrease
            return 0
        return self.train_losses[-2] - self.train_losses[-1]

    def add_validate_loss(self, epoch, new_validate_loss):
        self.epochs.append(epoch)
        self.validate_losses.append(new_validate_loss)

        # the decrease in validation loss since the lowest validation loss recorded (to avoid keeping list)
        self.validate_loss_decrease_thresh = 0 if self.lowest_validate_loss is None else self.lowest_validate_loss - new_validate_loss

        # update the lowest validation loss information
        if (self.lowest_validate_loss is None) or ((self.lowest_validate_loss - new_validate_loss) > self.mld):
            self.lowest_validate_loss = new_validate_loss
            self.num_epochs_since_lowest_validate_loss = 0
            self.epoch_with_lowest_validate_loss = epoch
        else:
            self.num_epochs_since_lowest_validate_loss += 1

    def get_validate_loss_decrease(self):
        # the decrease in validation loss since the last epoch
        if len(self.validate_losses) < 2:
            # if no losses have been reported or if only one loss has been reported, there is no loss decrease
            return 0
        return self.validate_losses[-2] - self.validate_losses[-1]


def run_training(data, log_dir, args):

    # reset the current graph and reset all the seeds before training
    tf.compat.v1.reset_default_graph()
    logger.info("setting random seeds py={}, np={}, tf={}".format(args.py_rseed, args.np_rseed, args.tf_rseed))
    random.seed(args.py_rseed)
    np.random.seed(args.np_rseed)
    tf.compat.v1.set_random_seed(args.tf_rseed)

    # set the encoded data to its own var to make things cleaner
    ed = data["encoded_data"]

    # build tensorflow computation graph -- do not reset the graph (make sure reset_graph=False) because
    # the tensorflow random seed set above is only for the default graph. if we reset the graph in this function call,
    # then the random seed will not longer apply
    igraph, tgraph = build_graph_from_args_dict(args, encoded_data_shape=ed["train"].shape, reset_graph=False)

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
        sess.run(tgraph["init_global"])

        loss_tracker = LossTracker(args.min_loss_decrease)

        # start the training loop
        logger.info("starting training loop")
        for epoch in range(1, args.epochs + 1):

            # flush stdout at the start of each epoch -- seems to help a bit with htcondor log files?
            sys.stdout.flush()

            # keep track of real time for this epoch
            epoch_start_time = time.time()

            # perform the training in batches (steps) for this epoch
            # this function will update the network weights and return how long it took to do so for each batch on avg
            avg_step_duration = run_training_epoch(sess, args, igraph, tgraph, data, epoch, step_display_interval)

            # end of the epoch - compute loss decrease on training set
            avg_train_loss = compute_loss(sess, igraph, tgraph, data, "train", args.batch_size)
            loss_tracker.add_train_loss(epoch, avg_train_loss)

            # end of epoch - compute loss on tune set to check for early stopping
            validate_loss = compute_loss(sess, igraph, tgraph, data, "tune", args.batch_size)
            loss_tracker.add_validate_loss(epoch, validate_loss)

            # duration statistics
            epoch_duration = time.time() - epoch_start_time

            print("====================")
            print("= Epoch: {:3}".format(epoch))
            print("= Duration: {:.2f}".format(epoch_duration))
            print("= Avg Step Duration: {:.4f}".format(avg_step_duration))
            print("= Training Loss: {:.6f}".format(avg_train_loss))
            print("= Training Loss Decrease (last epoch): {:.6f}".format(loss_tracker.get_train_loss_decrease()))
            print("= Validation Loss: {:.6f}".format(validate_loss))
            print("= Validation Loss Decrease (last epoch): {:.6f}".format(loss_tracker.get_validate_loss_decrease()))
            print("= Validation Loss Decrease (threshold): {:.6f}".format(loss_tracker.validate_loss_decrease_thresh))
            print("= Num Epochs Since Lowest Validation Loss: {}".format(loss_tracker.num_epochs_since_lowest_validate_loss))
            print("====================")

            # add per epoch summaries to tensorboard
            summary_str = sess.run(tgraph["summaries_per_epoch"],
                                   feed_dict={tgraph["validation_loss_ph"]: validate_loss,
                                              tgraph["training_loss_ph"]: avg_train_loss})
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
            met_early_stopping_allowance = loss_tracker.num_epochs_since_lowest_validate_loss == args.early_stopping_allowance

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
                print("Loading best model (epoch {}) and evaluating it.".format(loss_tracker.epoch_with_lowest_validate_loss))
                load_checkpoint(sess, saver, log_dir, loss_tracker.epoch_with_lowest_validate_loss)
                # we pass "None" as the epoch so that the evaluate function doesn't add this evaluation to TensorBoard
                evaluations = evaluate(sess, args, igraph, tgraph, None, data, ed.keys(), summary_writers)

                # save the evaluations
                save_metrics_evaluations(evaluations, log_dir, loss_tracker.epoch_with_lowest_validate_loss, early=True, args=args)

                clean_up_checkpoints(epoch, loss_tracker.epoch_with_lowest_validate_loss, log_dir,
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


def evaluations_dict_to_dataframe(evaluations, epoch=None, early=None):
    # process evaluations to remove the summary, predicted, and true scores
    p_evaluations = {}
    for set_name, evaluation in evaluations.items():
        evaluation = {metric_name: value for metric_name, value in evaluation.items()
                      if metric_name not in ["summary", "predicted", "true"]}
        if epoch is not None:
            evaluation["epoch"] = epoch
        if early is not None:
            evaluation["early"] = early
        p_evaluations[set_name] = evaluation

    # create a pandas dataframe of the evaluation metrics to save as a tsv
    sorted_order = ["train", "tune", "test", "stest"]
    metrics_df = pd.DataFrame(p_evaluations).transpose()
    metrics_df.index.rename("set", inplace=True)
    metrics_df = metrics_df.sort_index(
        key=lambda sets: [sorted_order.index(s) if s in sorted_order else len(sorted_order) for s in sets])
    return metrics_df


def save_metrics_evaluations(evaluations, log_dir, epoch, early, args):
    """ saves metrics for all evaluations """

    # save the evaluation metrics for all sets as a .tsv / .txt file
    metrics_df = evaluations_dict_to_dataframe(evaluations, epoch, early)
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
    log_dir_str = "log_{}_{}_{}_{}_{}_lr{}_bs{}_{}"

    # just use the net file basename
    net_arg = basename(args.net_file)[:-4]

    # dataset file basename if no dataset_name is specified
    if args.dataset_name != "":
        ds_arg = args.dataset_name
    else:
        ds_arg = basename(args.dataset_file)[:-4]

    format_args = [args.cluster, args.process, time.strftime("%Y-%m-%d_%H-%M-%S"),
                   ds_arg, net_arg, args.learning_rate, args.batch_size, shortuuid.encode(uuid.uuid4())[:8]]

    log_dir = join(args.log_dir_base, log_dir_str.format(*format_args))

    # log directory already exists. so just append a number to it.
    # should only happen if you run the script within the same second with the same args.
    # extra note: now that the log dir also includes a UUID, this *really* shouldn't happen
    if isdir(log_dir):
        log_dir = log_dir + "_2"
    while isdir(log_dir):
        log_dir = "_".join(log_dir.split("_")[:-1] + [str(int(log_dir.split("_")[-1]) + 1)])
        if not isdir(log_dir):
            break
    return log_dir


def main(args):
    """ set up params, log dir, splits, encode the data, and run the training """

    logger.info("software version {}".format(utils.__version__))

    # set up log directory & save the args file to it
    log_dir = log_dir_name(args)
    logger.info("log directory is {}".format(log_dir))
    utils.mkdir(log_dir)
    save_args(vars(args), join(log_dir, "args.txt"))

    # load the dataset
    if args.dataset_name != "":
        dataset_file = constants.DATASETS[args.dataset_name]["ds_fn"]
    else:
        dataset_file = args.dataset_file
    logger.info("loading dataset from {}".format(dataset_file))
    ds = utils.load_dataset(ds_fn=dataset_file)

    # load the dataset split or create one
    if args.split_dir != "":
        if isdir(args.split_dir):
            logger.info("loading split from {}".format(args.split_dir))
            split = sd.load_split_dir(args.split_dir)
            if isinstance(split, list):
                raise ValueError("this script doesn't support multiple reduced train size replicates in a single run. "
                                 "run each one individually by specifying the split dir of the replicate. ")
        else:
            raise FileNotFoundError("specified split dir doesn't exist: {}".format(args.split_dir))
    else:
        # create a classic train-tune-test split based on the specified args
        logger.info("creating a train/test split with tr={}, tu={}, and te={}, seed={}".format(
            args.train_size, args.tune_size, args.test_size, args.split_rseed
        ))
        split, _ = sd.train_tune_test(ds, train_size=args.train_size, tune_size=args.tune_size,
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
    parser = get_parser()
    parsed_args = parser.parse_args()
    if parsed_args.dataset_name == "":
        if parsed_args.dataset_file == "" or parsed_args.wt_aa == "" or parsed_args.wt_ofs == "":
            parser.error("you must specify either a dataset_name (for a dataset defined in constants.py) or "
                         "all three of the dataset_file, the wt_aa, and the wt_ofs. if you specify the dataset_name,"
                         "it takes priority over all the other args.")
    main(parsed_args)
