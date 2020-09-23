""" builds the tensorflow execution graph for the model by parsing network_specs """

import collections
from os.path import join, isfile, dirname
import argparse

import yaml
import tensorflow as tf

from my_pipgcn import node_average_gc
import gen_structure_graph as gsg


def eval_spec(spec, local_scope):
    # we can make this a bit safer by making an explicit scope list for eval() that only contains
    # allowable functions. but, as long as you're not running arbitrary yaml files through this it is fine
    if isinstance(spec, dict):
        return {k: eval_spec(v, local_scope) for k, v in spec.items()}
    elif isinstance(spec, list):
        return [eval_spec(i, local_scope) for i in spec]
    # string base case
    elif isinstance(spec, str) and spec.startswith("~"):
        return eval(spec[1:], globals(), local_scope)
    # all other non-iterable types base case
    return spec


def bg_inference(net_fn, adj_mtx, ph_inputs_dict):
    """ builds the graph as far as needed to return the tensor that would contain the output predictions """

    with open(net_fn, "r") as f:
        yml = yaml.safe_load(f)

    # a list to keep track of each layer, starting with the raw sequences input placeholder
    layers = [ph_inputs_dict["raw_seqs"]]

    for layer_spec in yml["network"]:
        # special check for presence of adjacency matrix for graph convolutional layer
        if layer_spec["layer_func"] == "~node_average_gc" and adj_mtx is None:
            raise ValueError("must specify a protein structure graph (adj_mtx) when using a graph convolutional layer")

        # eval the special vars/function names starting with "~"
        parsed_spec = eval_spec(layer_spec, locals())

        # create the layer and add to the list
        layer = parsed_spec["layer_func"](**parsed_spec["arguments"])
        layers.append(layer)

    # add the final output layer depending on how many tasks we are predicting
    num_tasks = 1
    layers.append(tf.layers.dense(layers[-1], units=num_tasks, activation=None, name="output"))
    predictions = tf.squeeze(layers[-1], axis=1)

    return predictions


def bg_loss(args, inf_graph):
    """ builds the graph by adding the required loss ops """
    # get tensorflow variables from the inference graph and other variables from args
    scores = inf_graph["ph_inputs_dict"]["scores"]
    predictions = inf_graph["predictions"]
    loss = tf.compat.v1.losses.mean_squared_error(labels=scores,
                                        predictions=predictions,
                                        reduction=tf.compat.v1.losses.Reduction.MEAN)
    return loss


def bg_training(args, loss):
    """ adds operations needed for training """
    learning_rate = args["learning_rate"]

    # create the adam descent optimizer with the given learning rate
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    # create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return global_step, train_op


def get_placeholder_inputs(data_shape):
    """ sets up the place holder input variables, into which the actual data will be fed """
    # remove the batch dimension from the data shape (this can be inferred real time and is not needed for
    # constructing the net)
    raw_seqs_shape = tuple([None] + list(data_shape[1:]))

    # put all placeholders in a dictionary
    ph_dict = {
        # placeholder for main input data
        "raw_seqs": tf.compat.v1.placeholder(tf.float32, shape=raw_seqs_shape, name="raw_seqs_placeholder"),
        # the score of the example (enrichment ratio, score from Enrich2)
        "scores": tf.compat.v1.placeholder(tf.float32, shape=None, name="scores_placeholder"),
        # placeholder for dropout
        "training": tf.compat.v1.placeholder_with_default(False, shape=(), name="training_ph")
    }
    return ph_dict


def get_placeholder_metrics():
    """ add placeholder for evaluation metrics to graph. these are used as a simple way to get the evaluation metrics
        added to the graph and tensorboard """
    # metrics computed outside of tensorflow and added to the graph for the tensorboard visualization
    mse_ph = tf.compat.v1.placeholder(tf.float32, name="mean_squared_error")
    pearsonr_ph = tf.compat.v1.placeholder(tf.float32, name="pearsonr")
    r2_ph = tf.compat.v1.placeholder(tf.float32, name="r2")
    spearmanr_ph = tf.compat.v1.placeholder(tf.float32, name="spearmanr")

    metrics_ph_dict = {"mse": mse_ph,
                       "pearsonr": pearsonr_ph,
                       "r2": r2_ph,
                       "spearmanr": spearmanr_ph}

    validation_loss_ph = tf.compat.v1.placeholder(tf.float32, name="validation_loss_placeholder")
    training_loss_ph = tf.compat.v1.placeholder(tf.float32, name="training_loss_placeholder")

    return metrics_ph_dict, validation_loss_ph, training_loss_ph


def bg_summaries(metrics_ph_dict, validation_loss_ph, training_loss_ph):
    """ add summary scalars to the graph for keeping track of various stats for tensorboard """
    # validation and training loss, evaluated at every epoch
    tf.compat.v1.summary.scalar("validation_loss", validation_loss_ph, collections=["summaries_per_epoch"])
    tf.compat.v1.summary.scalar("training_loss", training_loss_ph, collections=["summaries_per_epoch"])

    # metrics (not necessarily evaluated at every epoch, although I have been doing that)
    tf.compat.v1.summary.scalar("mse", metrics_ph_dict["mse"], collections=["summaries_metrics"])
    tf.compat.v1.summary.scalar("pearsonr", metrics_ph_dict["pearsonr"], collections=["summaries_metrics"])
    tf.compat.v1.summary.scalar("r2", metrics_ph_dict["r2"], collections=["summaries_metrics"])
    tf.compat.v1.summary.scalar("spearmanr", metrics_ph_dict["spearmanr"], collections=["summaries_metrics"])

    # build the summary Tensor based on the TF collection of summaries
    # this is used as the "op" to feed in values for the above metrics
    summaries_per_epoch = tf.compat.v1.summary.merge_all("summaries_per_epoch")
    summaries_metrics = tf.compat.v1.summary.merge_all("summaries_metrics")

    return summaries_per_epoch, summaries_metrics


def build_inference_graph(args, encoded_data_shape):
    """ builds the inference part of the graph. the encoded data shape is expected to have the first dimension
        be the number of examples (batch size). it will be ignored, but still expected, so make it 1 if needed """

    # load adjacency matrix for gcn
    graph_fn = args["graph_fn"]
    adj_mtx = None
    if isfile(graph_fn):
        # no need to throw an error if the file isn't found because we aren't sure if there are graph convolutional
        # layers in this network. An error will be thrown later if there are graph layers and no adj_mtx was specified
        g = gsg.load_graph(graph_fn)
        adj_mtx = gsg.ordered_adjacency_matrix(g)
        # adjacency matrices will be made part of the actual graph rather than feeding them in with feed_dict
        # this will make it so there's no need to re-load the adjacency matrices when loading model from checkpoint
        adj_mtx = tf.convert_to_tensor(adj_mtx)

    # placeholder for inputs (raw sequences, labels, weights, is_training, etc)
    ph_inputs_dict = get_placeholder_inputs(encoded_data_shape)

    # build the inference part of the graph that gets the output values from the inputs
    predictions = bg_inference(args["net_file"], adj_mtx, ph_inputs_dict)

    # place all relevant tensorflow variables and ops into a dictionary for passing around to other functions
    inf_graph = {"ph_inputs_dict": ph_inputs_dict,
                 "predictions": predictions}

    return inf_graph


def build_training_graph(args, inf_graph):
    """ builds the training part of the graph """

    # placeholders for metrics and validation, training loss
    metrics_ph_dict, validation_loss_ph, training_loss_ph = get_placeholder_metrics()
    summaries_per_epoch, summaries_metrics = bg_summaries(metrics_ph_dict, validation_loss_ph, training_loss_ph)

    # op for loss calculation
    loss = bg_loss(args, inf_graph)

    # ops that calculate and apply gradients as well as global step
    global_step, train_op = bg_training(args, loss)

    # variable initializer op for initializing network weights
    init_global = tf.compat.v1.global_variables_initializer()

    train_graph = {"loss": loss,
                   "global_step": global_step,
                   "train_op": train_op,
                   "init_global": init_global,
                   "summaries_per_epoch": summaries_per_epoch,
                   "summaries_metrics": summaries_metrics,
                   "validation_loss_ph": validation_loss_ph,
                   "training_loss_ph": training_loss_ph,
                   "metrics_ph_dict": metrics_ph_dict}

    return train_graph


def build_graph_from_args_dict(args, encoded_data_shape, reset_graph=True):
    # if args was processed with argparse, it will be a namespace object, but we are set up to work on a dict
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    if reset_graph:
        tf.compat.v1.reset_default_graph()

    inf_graph = build_inference_graph(args, encoded_data_shape)
    train_graph = build_training_graph(args, inf_graph)

    return inf_graph, train_graph


def main():
    pass


if __name__ == "__main__":
    main()
