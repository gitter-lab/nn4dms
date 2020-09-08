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
import tensorflow as tf

# mine
import utils
import split_dataset as sd
from build_tf_model import build_graph_from_args_dict

logger = logging.getLogger("nn4dms." + __name__)
logger.setLevel(logging.INFO)


def compute_loss(sess, igraph, tgraph, data, edge_data, scores, weights, batch_size):
    """ computes the average loss over all data batches """

    edge_data_ph = igraph["ph_inputs_dict"]["edge_data"]
    raw_seqs_ph = igraph["ph_inputs_dict"]["raw_seqs"]
    scores_ph = igraph["ph_inputs_dict"]["scores"]
    weights_ph = igraph["ph_inputs_dict"]["weights"]
    loss = tgraph["loss"]

    if edge_data_ph is None:
        bg = utils.batch_generator((data, scores, weights), batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)
    else:
        bg = utils.batch_generator((data, edge_data, scores, weights), batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)

    loss_vals = []
    for batch_num, batch_data in enumerate(bg):

        if edge_data_ph is None:
            data_batch, scores_batch, weights_batch = batch_data
        else:
            data_batch, edge_data_batch, scores_batch, weights_batch = batch_data

        # fill the feed dict with the next batch
        feed_dict = {raw_seqs_ph: data_batch, scores_ph: scores_batch, weights_ph: weights_batch}
        if edge_data_ph is not None:
            feed_dict[edge_data_ph] = edge_data_batch

        # get predicted labels for evaluating metrics using sklearn
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vals.append(loss_val)

    return np.average(loss_vals)


def run_eval(sess, args, igraph, tgraph, ds, idxs, data, edge_data, scores, weights):
    """ runs one evaluation against the full epoch of data """

    # get variables to make transition to new igraph, tgraph system easier
    edge_data_ph = igraph["ph_inputs_dict"]["edge_data"]
    raw_seqs_ph = igraph["ph_inputs_dict"]["raw_seqs"]
    scores_ph = igraph["ph_inputs_dict"]["scores"]
    predictions = igraph["predictions"]
    metrics_ph_dict = tgraph["metrics_ph_dict"]
    summaries_metrics = tgraph["summaries_metrics"]

    if edge_data_ph is None:
        bg = utils.batch_generator((data, scores, weights),
                                   args.batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)
    else:
        bg = utils.batch_generator((data, edge_data, scores, weights),
                                   args.batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)

    # get all the predicted and true labels in batches
    predicted_scores = np.zeros(scores.shape)
    true_scores = np.zeros(scores.shape)

    start = time.time()
    for batch_num, batch_data in enumerate(bg):

        if edge_data_ph is None:
            data_batch, scores_batch, weights_batch = batch_data
        else:
            data_batch, edge_data_batch, scores_batch, weights_batch = batch_data

        # fill the feed dict with the next batch
        feed_dict = {raw_seqs_ph: data_batch, scores_ph: scores_batch}
        if edge_data_ph is not None:
            feed_dict[edge_data_ph] = edge_data_batch

        # start and end index for this batch
        start_index = batch_num * args.batch_size
        end_index = start_index + args.batch_size

        # get predicted labels for evaluating metrics using sklearn
        preds = sess.run(predictions, feed_dict=feed_dict)
        predicted_scores[start_index:end_index] = preds
        true_scores[start_index:end_index] = scores_batch

    duration = time.time() - start

    # if we are doing multitask learning, extract just the fitness score (first column)
    if len(true_scores.shape) > 1 and true_scores.shape[1] > 1:
        true_scores = true_scores[:, 0]
        predicted_scores = predicted_scores[:, 0]

    evaluation_dict = regression_metrics.compute_metrics(true_scores, predicted_scores, ds, idxs)

    # update summaries by running metrics ops
    if summaries_metrics is not None:

        metrics_feed_dict = {metrics_ph_dict["mse"]: evaluation_dict["mse"],
                             metrics_ph_dict["r"]: evaluation_dict["r"],
                             metrics_ph_dict["r2"]: evaluation_dict["r2"],
                             metrics_ph_dict["r_singles"]: evaluation_dict["r_singles"],
                             metrics_ph_dict["r_multi"]: evaluation_dict["r_multi"],
                             metrics_ph_dict["r_ep"]: evaluation_dict["r_ep"],
                             metrics_ph_dict["ev"]: evaluation_dict["ev"]}

        out = sess.run([summaries_metrics,
                        metrics_ph_dict["mse"],
                        metrics_ph_dict["r"],
                        metrics_ph_dict["r2"],
                        metrics_ph_dict["r_singles"],
                        metrics_ph_dict["r_multi"],
                        metrics_ph_dict["r_ep"],
                        metrics_ph_dict["ev"]],
                       feed_dict=metrics_feed_dict)

        summary_str = out[0]
    else:
        summary_str = None

    # add summary string to evaluation dict
    evaluation_dict["summary"] = summary_str

    # print
    print("  Evaluation completed in {:.3} sec.".format(duration))
    print("  MSE: {:.3f}".format(evaluation_dict["mse"]))
    print("  r: {:.3f}".format(evaluation_dict["r"]))
    print("  r_singles: {:.3f}".format(evaluation_dict["r_singles"]))
    print("  r_multi: {:.3f}".format(evaluation_dict["r_multi"]))
    print("  r_ep: {:.3f}".format(evaluation_dict["r_ep"]))
    print("  r_ep_pos: {:.3f}".format(evaluation_dict["r_ep_pos"]))
    print("  r_ep_neg: {:.3f}".format(evaluation_dict["r_ep_neg"]))
    print("  r2: {:.3f}".format(evaluation_dict["r2"]))
    print("  ev: {:.3f}".format(evaluation_dict["ev"]))

    return evaluation_dict


def evaluate(sess, args, igraph, tgraph, epoch, ds, split_idxs, data_sets, edge_data_sets,
             scores_sets, weights_sets, summary_writers, evaluate_test_set=False, evaluate_super_test=False):
    """ perform evaluation on the training & validation sets, printing output & saving to summary writer
        setting evaluate_test_set to true will only evaluate the test set if it exists """

    sorted_set_names = ["train", "validate", "test", "super_test"]

    # the sets to evaluate
    sets = list(split_idxs)
    if "test" in sets and not evaluate_test_set:
        sets.remove("test")
    if "super_test" in sets and not evaluate_super_test:
        sets.remove("super_test")

    # dictionary to store results of evaluations
    evaluations = {}

    for set_name in sorted(sets, key=lambda x: sorted_set_names.index(x)):
        print("Evaluation: {}".format(set_name))
        evaluations[set_name] = run_eval(sess, args, igraph, tgraph, ds,
                                         split_idxs[set_name],
                                         data_sets[set_name],
                                         edge_data_sets[set_name],
                                         scores_sets[set_name],
                                         weights_sets[set_name])

        if epoch and (set_name == "train" or set_name == "validate"):
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


def run_training(ds, split_idxs, data_sets, edge_data_sets, scores_sets, weights_sets, log_dir, args):

    # reset the current graph and reset all the seeds before
    tf.reset_default_graph()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # build tensorflow computation graph
    # note: could update build_graph_from_args_dict to take in just the data shape instead of the full data
    igraph, tgraph = build_graph_from_args_dict(args, reset_graph=False,
                                                encoded_data=data_sets["train"],
                                                encoded_edge_data=edge_data_sets["train"])

    # convert dictionary entries to variables to make transition easier
    init_global = tgraph["init_global"]
    edge_data_ph = igraph["ph_inputs_dict"]["edge_data"]
    train_op = tgraph["train_op"]
    loss = tgraph["loss"]
    summaries_per_epoch = tgraph["summaries_per_epoch"]
    validation_loss_ph = tgraph["validation_loss_ph"]
    training_loss_ph = tgraph["training_loss_ph"]

    # get the step display interval
    step_display_interval = get_step_display_interval(args, len(data_sets["train"]))

    # Create a saver for writing training checkpoints.
    max_to_keep = args.early_stopping_allowance + 1 if args.early_stopping else 2
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=max_to_keep)

    # tf.get_default_graph().finalize()

    # create a session for running Ops on the Graph.
    sess_config = tf.ConfigProto(
                    intra_op_parallelism_threads=args.num_threads,
                    log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary_writers = {"train": tf.summary.FileWriter(join(log_dir, "train")),
                           "validate": tf.summary.FileWriter(join(log_dir, "validation"))}

        # Run the Op to initialize the variables.
        sess.run(init_global)

        # keep track of when lowest loss... will be used for stopping criteria
        epoch_with_lowest_validate_loss = 1
        lowest_validate_loss = None
        validate_loss_last_epoch = None
        train_loss_last_epoch = None
        num_epochs_since_lowest_validate_loss = 0

        logging.info("starting training loop")
        # Start the training loop.
        for epoch in range(1, args.epochs + 1):

            # flush stdout at the start of each epoch
            sys.stdout.flush()

            epoch_start_time = time.time()

            # keep some statistics for this epoch
            epoch_step_durations = []
            epoch_train_loss_values = []

            # keep some statistics for this step interval
            start_step = 1
            interval_step_durations = []
            interval_train_loss_values = []

            # generate the data batches
            if edge_data_ph is None:
                bg = utils.batch_generator((data_sets["train"], scores_sets["train"], weights_sets["train"]),
                                       args.batch_size, skip_last_batch=False, num_epochs=1)
            else:
                bg = utils.batch_generator((data_sets["train"], edge_data_sets["train"], scores_sets["train"], weights_sets["train"]),
                                           args.batch_size, skip_last_batch=False, num_epochs=1)

            for step, batch_data in enumerate(bg):
                if edge_data_ph is None:
                    data_batch, scores_batch, weights_batch = batch_data
                    edge_data_batch = None
                else:
                    data_batch, edge_data_batch, scores_batch, weights_batch = batch_data

                step += 1
                # logging.info("starting step {}".format(step))
                step_start_time = time.time()

                # create the feed dictionary to feed batch inputs into the graph
                feed_dict = {igraph["ph_inputs_dict"]["raw_seqs"]: data_batch,
                             igraph["ph_inputs_dict"]["scores"]: scores_batch,
                             igraph["ph_inputs_dict"]["weights"]: weights_batch,
                             igraph["ph_inputs_dict"]["training"]: True}

                # print(sess.run(tf.shape(predictions), feed_dict=feed_dict))
                # print(sess.run(tf.shape(scores_ph), feed_dict=feed_dict))
                # print(sess.run(scores_ph, feed_dict=feed_dict))

                # add edge data batch to the feed dict if it is given
                if edge_data_ph is not None:
                    feed_dict[edge_data_ph] = edge_data_batch

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
            # avg_train_loss = np.average(epoch_train_loss_values)
            avg_train_loss = compute_loss(sess, igraph, tgraph, data_sets["train"], edge_data_sets["train"],
                                          scores_sets["train"], weights_sets["train"], args.batch_size)
            train_loss_decrease_last_epoch = 0 if train_loss_last_epoch is None else train_loss_last_epoch - avg_train_loss
            train_loss_last_epoch = avg_train_loss

            # end of epoch - compute loss on validation set to check for early stopping
            validate_loss = compute_loss(sess, igraph, tgraph, data_sets["validate"], edge_data_sets["validate"],
                                         scores_sets["validate"], weights_sets["validate"], args.batch_size)
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
            summary_str = sess.run(summaries_per_epoch, feed_dict={validation_loss_ph: validate_loss, training_loss_ph: avg_train_loss})
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()

            # save a checkpoint periodically or if it's the last epoch
            if epoch % args.epoch_checkpoint_interval == 0 or epoch == args.epochs:
                save_checkpoint(sess, saver, log_dir, epoch)

            # evaluate the model periodically, or if it's the last epoch
            if epoch % args.epoch_evaluation_interval == 0 or epoch == args.epochs:
                # print("NOT EARLY STOPPING BUT RE-LOADING CHECKPOINT.....".format(epoch_with_lowest_validate_loss))
                # # added: reinitalize all variables before reloading checkpoint
                # print("Re-initializing all variables before loading checkpoint...")
                # tf.set_random_seed(7)
                # sess.run(init_global)
                # load_checkpoint(sess, saver, log_dir, epoch_with_lowest_validate_loss)

                evaluations = evaluate(sess, args, igraph, tgraph, epoch, ds, split_idxs,
                                       data_sets, edge_data_sets, scores_sets, weights_sets,
                                       summary_writers, evaluate_test_set=(epoch == args.epochs),
                                       evaluate_super_test=True)

                # hit the last epoch, save its evaluation
                if epoch == args.epochs:
                    save_metrics_evaluations(evaluations, ds, split_idxs, log_dir, epoch, weights_sets, early=False, args=args)

                    clean_up_checkpoints(epoch, epoch, log_dir, delete_checkpoints=args.delete_checkpoints, compress_checkpoints=False)
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
                if epoch % args.epoch_checkpoint_interval != 0 and epoch != args.epochs:
                    save_checkpoint(sess, saver, log_dir, epoch)

                # if we didn't already evaluate this epoch, evaluate now
                if epoch % args.epoch_evaluation_interval != 0 and epoch != args.epochs:
                    evaluate(sess, args, igraph, tgraph, epoch, ds, split_idxs, data_sets, edge_data_sets,
                             scores_sets, weights_sets, summary_writers)

                # load the best model and evaluate it
                print("Loading best model (epoch {}) and evaluating it.".format(epoch_with_lowest_validate_loss))
                # added: reinitalize all variables before reloading checkpoint
                # print("Re-initializing all variables before loading checkpoint...")
                # sess.run(init_global)
                load_checkpoint(sess, saver, log_dir, epoch_with_lowest_validate_loss)

                evaluations = evaluate(sess, args, igraph, tgraph, None, ds, split_idxs,  data_sets, edge_data_sets,
                                       scores_sets, weights_sets, summary_writers,
                                       evaluate_test_set=True, evaluate_super_test=True)

                save_metrics_evaluations(evaluations, ds, split_idxs, log_dir, epoch_with_lowest_validate_loss, weights_sets, early=True, args=args)

                clean_up_checkpoints(epoch, epoch_with_lowest_validate_loss, log_dir, delete_checkpoints=args.delete_checkpoints, compress_checkpoints=False)
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

    # np.savez_compressed("{}_scores.npz".format(fn_base), predicted=evaluation["predicted"], true=evaluation["true"])

    np.save("{}_predicted_scores.npy".format(fn_base), evaluation["predicted"])
    np.save("{}_true_scores.npy".format(fn_base), evaluation["true"])

    # np.savetxt("{}_predicted_scores.txt".format(fn_base), evaluation["predicted"], fmt="%f")
    # np.savetxt("{}_true_scores.txt".format(fn_base), evaluation["true"], fmt="%f")


def save_metrics_evaluations(evaluations, ds, split_idxs, log_dir, epoch, weights_sets, early, args):
    """ saves metrics for all evaluations """

    sorted_set_names = ["train", "validate", "test", 'super_test']
    for set_name in sorted(evaluations, key=lambda x: sorted_set_names.index(x)):
        save_metrics(set_name, ds, split_idxs, log_dir, epoch, evaluations[set_name], weights_sets, early, args)


def save_metrics(set_name, ds, split_idxs, log_dir, epoch, evaluation, weights_sets, early, args):

    with open(join(log_dir, "final_evaluation.txt"), "a+") as f:
        f.write("----- Eval: {} -----\n".format(set_name))
        f.write("Epoch: {} ({})\n".format(epoch, "Early" if early else "Max"))
        for metric_name in sorted(evaluation):
            metric_val = evaluation[metric_name]
            if metric_name not in ["summary", "predicted", "true"]:
                f.write("{}: {}\n".format(metric_name, metric_val))

    # plot regression scores vs. actual scores
    out_dir = join(log_dir, "graphs")
    utils.ensure_dir_exists(out_dir)

    # set the plot style... this is not really as important anymore since these kinds of plots are now plotted
    # after the models are trained. The point of plotting during training is to just get a quick look
    # this can really be cleaned up a lot...
    if args is None:
        train_plot_style = PlotStyle.NONE
        valid_plot_style = PlotStyle.NONE
    elif args.train_style == TrainStyle.TRAIN_TEST:
        train_plot_style = PlotStyle.SINGLE_MULTI_SEP
        valid_plot_style = PlotStyle.SINGLE_MULTI_SEP
    else:
        train_plot_style = PlotStyle.NONE
        valid_plot_style = PlotStyle.NONE

    if set_name == "train":
        plot_style = train_plot_style
    else:
        plot_style = valid_plot_style

    split_idxs_for_set = split_idxs[set_name] if split_idxs is not None else None

    plot_scores(ds, split_idxs_for_set, evaluation, set_name, join(out_dir, "{}_scores".format(set_name)),
                weights=weights_sets[set_name], plot_style=plot_style)

    # save true scores and actual scores predicted using this model
    out_dir = join(log_dir, "predictions")
    utils.ensure_dir_exists(out_dir)

    save_scores(evaluation, join(out_dir, set_name))


def save_checkpoint(sess, saver, log_dir, epoch):

    checkpoint_file = join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_file, global_step=epoch)


def load_checkpoint(sess, saver, log_dir, epoch):

    checkpoint_file = join(log_dir, "model.ckpt-{}".format(epoch))
    # print_tensors_in_checkpoint_file(checkpoint_file, all_tensors=True, tensor_name='')
    saver.restore(sess, checkpoint_file)


def parse_argument_file(arg_file):
    """ parses the given argument file and generates an argparse string for it """

    args = []
    with open(arg_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            args_line = [val.strip() for val in line.split(":")]
            args.append("--{}".format(args_line[0]))
            args.append(args_line[1])

    return args


def get_log_dir(args):

    if args.log_dir_style == "gb1_resampling":
        # cluster, process, parsed ds_fn, net type, learning rate, batch size, date
        log_dir_str = "nn-{}-{}-{}-net_{}-lr_{}-bs_{}-{}"
        p_ds_fn = os.path.basename(args.ds_fn)[:-4]
        p_net = args.net_file.split("/")[-1][4:-5]
        format_args = [args.cluster, args.process, p_ds_fn, p_net, args.learning_rate, args.batch_size, time.strftime("%m%d%y-%H%M%S")]
        log_dir = join(args.log_dir_base, log_dir_str.format(*format_args))
    else:
        if args.train_style == "lopo":
            log_dir_str = "lopo-{}-{}-{}-et_{}-net_{}-enc_{}-ts_{}-lr_{}-{}"
        else:
            log_dir_str = "nn-{}-{}-{}-et_{}-net_{}-enc_{}-eenc_{}-ts_{}-lr_{}-{}"

        net_arg = args.net_file.split("/")[-1][4:-5]

        if args.ds_fn.endswith(".txt"):
            ds_fn = os.path.basename(args.ds_fn)[:-4]
        else:
            ds_fn = os.path.basename(args.ds_fn)[:-4].split("_")[0]

        format_args = [args.cluster, args.process, ds_fn, args.error_thresh, net_arg, args.encodings, args.edge_encodings,
                       args.train_size, args.learning_rate, time.strftime("%m%d%y-%H%M%S")]
        log_dir = join(args.log_dir_base, log_dir_str.format(*format_args))

    return log_dir


def save_reduced_results(evaluations_all_splits, log_dir):

    # combine the metrics from each split into the corresponding dictionary
    combined_evaluations = {}

    for evaluations_for_split in evaluations_all_splits:
        # loop through the evaluation for each randomized training split

        for set_name, evaluation_for_set in evaluations_for_split.items():
            # loop through each set (train, validate, test)

            # if set is not in combined_evaluations, add it now
            if set_name not in combined_evaluations:
                combined_evaluations[set_name] = collections.defaultdict(list)

            for metric_name in sorted(evaluation_for_set):
                metric_val = evaluation_for_set[metric_name]
                if metric_name not in ["summary", "predicted", "true"]:
                    combined_evaluations[set_name][metric_name].append(metric_val)

    # save the combined evaluations
    with open(join(log_dir, "combined_results.txt"), "w") as f:
        f.write("COMBINED RESULTS\n")

        for set_name in combined_evaluations:
            f.write("\nSet: {}\n".format(set_name))

            for metric_name in sorted(combined_evaluations[set_name]):
                metric_vals = combined_evaluations[set_name][metric_name]
                f.write("\n{}: {}\n".format(metric_name, metric_vals))
                f.write("{} mean: {}\n".format(metric_name, np.mean(metric_vals)))
                f.write("{} median: {}\n".format(metric_name, np.median(metric_vals)))


def get_splits(ds, args):
    """ returns the split-idxs or the multiple splits for training in various FilterStyles"""

    # train-test
    if args.train_style == TrainStyle.TRAIN_TEST:
        if args.filter_style == FilterStyle.TRAIN_ALL_PREDICT_ALL:
            split_idxs = split_datasets.train_test(ds, train_size=args.train_size, test=args.use_test_set,
                                                   super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                   singles_only=False, singles_doubles_only=False, ep_only=False,
                                                   random_seed=RANDOM_SEED, threshold=args.error_thresh, fraction=1)

        elif args.filter_style == FilterStyle.TRAIN_SD_PREDICT_SD:
            split_idxs = split_datasets.train_test(ds, train_size=args.train_size, test=args.use_test_set,
                                                   super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                   singles_only=False, singles_doubles_only=True, ep_only=False,
                                                   random_seed=RANDOM_SEED, threshold=args.error_thresh, fraction=1)

        elif args.filter_style == FilterStyle.TRAIN_SINGLE_PREDICT_SINGLE:
            split_idxs = split_datasets.train_test(ds, train_size=args.train_size, test=args.use_test_set, singles_only=True,
                                                   super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                   singles_doubles_only=False, ep_only=False, random_seed=RANDOM_SEED,
                                                   threshold=args.error_thresh, fraction=1)

        elif args.filter_style == FilterStyle.TRAIN_EP_PREDICT_EP:
            split_idxs = split_datasets.train_test(ds, train_size=args.train_size, test=args.use_test_set, singles_only=False,
                                                   super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                   singles_doubles_only=False, ep_only=True, random_seed=RANDOM_SEED,
                                                   threshold=args.error_thresh, fraction=1)

        return split_idxs

    # reduced ds size
    elif args.train_style == TrainStyle.REDUCED_DS_SIZE:
        # load split indices for reduced ds size
        if args.filter_style == FilterStyle.TRAIN_ALL_PREDICT_ALL:
            splits = split_datasets.reduced_ds_size(ds, test=args.use_test_set, val_test_size=args.held_out_size,
                                                    super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                    train_prop=args.train_size, num_train_samples=args.num_train_samples,
                                                    singles_only=False, singles_doubles_only=False, ep_only=False,
                                                    threshold=args.error_thresh, fraction=1, random_seed=RANDOM_SEED)

        elif args.filter_style == FilterStyle.TRAIN_SD_PREDICT_SD:
            splits = split_datasets.reduced_ds_size(ds, test=args.use_test_set, val_test_size=args.held_out_size,
                                                    super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                    train_prop=args.train_size, num_train_samples=args.num_train_samples,
                                                    singles_only=False, singles_doubles_only=True, ep_only=False,
                                                    threshold=args.error_thresh, fraction=1, random_seed=RANDOM_SEED)

        elif args.filter_style == FilterStyle.TRAIN_SINGLE_PREDICT_SINGLE:
            splits = split_datasets.reduced_ds_size(ds, test=args.use_test_set, val_test_size=args.held_out_size,
                                                    super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                    train_prop=args.train_size, num_train_samples=args.num_train_samples,
                                                    singles_only=True, singles_doubles_only=False, ep_only=False,
                                                    threshold=args.error_thresh, fraction=1, random_seed=RANDOM_SEED)

        elif args.filter_style == FilterStyle.TRAIN_EP_PREDICT_EP:
            splits = split_datasets.reduced_ds_size(ds, test=args.use_test_set, val_test_size=args.held_out_size,
                                                    super_test=args.use_super_test_set, super_test_size=args.super_test_size,
                                                    train_prop=args.train_size, num_train_samples=args.num_train_samples,
                                                    singles_only=False, singles_doubles_only=False, ep_only=True,
                                                    threshold=args.error_thresh, fraction=1, random_seed=RANDOM_SEED)

        return splits



def run_train_test_model(args, log_dir):

    # load data
    ds, encoded_data, encoded_edge_data = utils.load_dataset_new(args.ds_fn, encs=args.encodings,
                                                                 edge_encs=args.edge_encodings,
                                                                 equal_weights=args.equal_weights)
    # load split indices
    if args.split_dir is not None:
        if args.split_dir_type == "relative":
            split_dir = join(dirname(args.ds_fn), args.split_dir)
        else:
            split_dir = args.split_dir
        logging.info("loading train/test split from {}".format(split_dir))
        split_idxs = utils.load_split_idxs(split_dir)
    else:
        # no split dir was specified, so generate split indices according to the other options in the config
        split_idxs = get_splits(ds, args)

    # save the split indices that are going to be used for this model to the log directory for the model
    utils.save_split_idxs(split_idxs, join(log_dir, "train_test_split"), npy=False)
    split_datasets.split_stats(ds, split_idxs)

    # create split sets (including loading up actual data and creating set for it)
    data_sets = utils.generate_split(encoded_data, split_idxs, load_test_set=True)

    # if "casp3" in args.ds_dir:
    #     for set_name, set in data_sets.items():
    #         data_sets[set_name] = set[:, 1:-9, ...]

    if len(encoded_edge_data) == 0:
        edge_data_sets = collections.defaultdict(lambda: None)
    else:
        edge_data_sets = utils.generate_split(encoded_edge_data[0], split_idxs, load_test_set=True)

    # load and create create target (score) sets based on given targets
    scores = select_target_scores(args, ds)
    scores_sets = utils.generate_split(scores, split_idxs, load_test_set=True)

    # load weights... these haven't been used
    weights_sets = utils.generate_split(ds["weight"].values, split_idxs, load_test_set=True)

    # log some info about the dataset
    for set_name in data_sets:
        logging.info("data {} shape: {}".format(set_name, data_sets[set_name].shape))

    evaluations = run_training(ds, split_idxs, data_sets, edge_data_sets, scores_sets, weights_sets, log_dir, args)


def run_reduced_model(args, log_dir):
    # load data
    ds, encoded_data, encoded_edge_data = utils.load_dataset_new(args.ds_fn, encs=args.encodings,
                                                                 edge_encs=args.edge_encodings,
                                                                 equal_weights=args.equal_weights)

    # load split indices for reduced ds size
    # todo: add support for loading splits from file (regular train-test already supported)
    splits = get_splits(ds, args)

    results = []
    for split_num, split_idxs in enumerate(splits):

        split_datasets.split_stats(ds, split_idxs)

        # load a NEW net spec for each model because during computation graph generation, the net spec is modified
        # in place! Ideally, change it so net is NOT modified in place. but for now this will work.
        with open(args.net_file, "r") as fp:

            # create split sets (including loading up actual data and creating set for it)
            data_sets = utils.generate_split(encoded_data, split_idxs, load_test_set=True)

            if len(encoded_edge_data) == 0:
                edge_data_sets = collections.defaultdict(lambda: None)
            else:
                edge_data_sets = utils.generate_split(encoded_edge_data, split_idxs, load_test_set=True)

            # load and create create target (score) sets based on given targets
            scores = select_target_scores(args, ds)
            scores_sets = utils.generate_split(scores, split_idxs, load_test_set=True)

            # load weights... these haven't been used
            weights_sets = utils.generate_split(ds["weight"].values, split_idxs, load_test_set=True)

            # log some info about the dataset
            for set_name in data_sets:
                logging.info("data {} shape: {}".format(set_name, data_sets[set_name].shape))

            # create a log directory for this training set subsample
            log_dir_fold = join(log_dir, "train-subsample-{}".format(split_num))
            utils.ensure_dir_exists(log_dir_fold)
            save_args_file(vars(args), log_dir_fold)
            utils.save_split_idxs(split_idxs, join(log_dir_fold, "train_test_split"), npy=False)

            evaluations = run_training(ds, split_idxs, data_sets, edge_data_sets, scores_sets, weights_sets, args)
            results.append(evaluations)

    save_reduced_results(results, log_dir)


def log_dir_name(args):
    # log directory captures the cluster & process (if running on HTCondor), the dataset name, the
    # network specification file basename, the learning rate, the batch size, and the date and time
    log_dir_str = "log_{}_{}_d-{}_n-{}_lr-{}_bs-{}_{}"

    # just use the arg file basename
    net_arg = basename(args.net_file)[4:-5]

    # dataset file basename
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
                if v != False:
                    f.write("--{}\n".format(k))
                    # if a flag is true, no need to specify the "true" value
                    if v != True:
                        f.write("{}\n".format(v))


def set_up_log_dir(args):
    log_dir = log_dir_name(args)
    logger.info("log directory is {}".format(log_dir))
    utils.mkdir(log_dir)
    save_args(vars(args), log_dir)
    return log_dir


def main(args):
    """ set up params, log dir and make call to run_training """

    # set up log directory & save the args file to it
    log_dir = set_up_log_dir(args)

    # load the dataset split or create one
    if isdir(args.split_dir):
        logger.info("loading dataset split from {}".format(args.split_dir))


    # load data
    ds, encoded_data, encoded_edge_data = utils.load_dataset_new(args.ds_fn, encs=args.encodings,
                                                                 edge_encs=args.edge_encodings,
                                                                 equal_weights=args.equal_weights)
    # load split indices
    if args.split_dir is not None:
        if args.split_dir_type == "relative":
            split_dir = join(dirname(args.ds_fn), args.split_dir)
        else:
            split_dir = args.split_dir
        logging.info("loading train/test split from {}".format(split_dir))
        split_idxs = utils.load_split_idxs(split_dir)
    else:
        # no split dir was specified, so generate split indices according to the other options in the config
        split_idxs = get_splits(ds, args)

    # save the split indices that are going to be used for this model to the log directory for the model
    utils.save_split_idxs(split_idxs, join(log_dir, "train_test_split"), npy=False)
    split_datasets.split_stats(ds, split_idxs)

    # create split sets (including loading up actual data and creating set for it)
    data_sets = utils.generate_split(encoded_data, split_idxs, load_test_set=True)

    # if "casp3" in args.ds_dir:
    #     for set_name, set in data_sets.items():
    #         data_sets[set_name] = set[:, 1:-9, ...]

    if len(encoded_edge_data) == 0:
        edge_data_sets = collections.defaultdict(lambda: None)
    else:
        edge_data_sets = utils.generate_split(encoded_edge_data[0], split_idxs, load_test_set=True)

    # load and create create target (score) sets based on given targets
    scores = select_target_scores(args, ds)
    scores_sets = utils.generate_split(scores, split_idxs, load_test_set=True)

    # load weights... these haven't been used
    weights_sets = utils.generate_split(ds["weight"].values, split_idxs, load_test_set=True)

    # log some info about the dataset
    for set_name in data_sets:
        logging.info("data {} shape: {}".format(set_name, data_sets[set_name].shape))

    evaluations = run_training(ds, split_idxs, data_sets, edge_data_sets, scores_sets, weights_sets, log_dir, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')

    # defining the dataset and network to train
    parser.add_argument("--dataset_name",
                        help="the name of the dataset. this dataset must be defined in constants.py",
                        type=str,
                        default="avgfp")

    parser.add_argument("--dataset_file",
                        help="the name of the dataset. this dataset must be defined in constants.py",
                        type=str,
                        default="data/avgfp/avgfp.tsv")

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

    # parser.add_argument("--supertest_size",
    #                     help="size of the held out supertest set",
    #                     type=float,
    #                     default=0.6)

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

    parser.add_argument("--test_prop",
                        help="proportion of train test for reduced training split",
                        type=float,
                        default=0.5)

    # parser.add_argument("--num_train_samples",
    #                     help="for reduced_ds_size, the number of different training samples",
    #                     type=int,
    #                     default=5)

    # miscellaneous training, printing, and cleanup arguments
    parser.add_argument("--step_display_interval",
                        help="display step interval",
                        type=float,
                        default=0.1)

    parser.add_argument("--epoch_checkpoint_interval",
                        help="checkpoint epoch step",
                        type=int,
                        default=5)

    parser.add_argument("--epoch_evaluation_interval",
                        help="perform evaluation on all splits every x epochs",
                        type=int,
                        default=5)

    # parser.add_argument("--train_style",
    #                     type=lambda train_style: TrainStyle[train_style],
    #                     choices=list(TrainStyle),
    #                     default="TRAIN_TEST")

    # parser.add_argument('--shuffle_scores',
    #                     type=str2bool,
    #                     default=False)

    parser.add_argument("--delete_checkpoints",
                        help="set this flag to delete checkpoints",
                        action="store_true")

    parser.add_argument("--compress_everything",
                        help="set flag to compress all output except args.txt and final_evaluation.txt",
                        action="store_true")

    parser.add_argument("--num_threads",
                        help="maximum number of threads to use",
                        type=int,
                        default=8)

    parser.add_argument("--log_dir_style",
                        help="log directory style type",
                        type=str,
                        default="regular",
                        choices=["regular", "gb1_resampling"])

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
    main(parsed_args)
