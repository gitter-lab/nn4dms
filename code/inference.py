""" use trained models to get predictions for new variants """

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import batch_generator


def restore_sess(ckpt_prefix_fn):
    """ create a TensorFlow session with restored parameters """

    # create a fresh graph for use with this session
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(ckpt_prefix_fn + ".meta", clear_devices=True)

    # set up the session and restore the parameters
    sess = tf.compat.v1.Session(graph=graph)
    saver.restore(sess, ckpt_prefix_fn)

    return sess


def run_inference(encoded_data, ckpt_prefix_fn=None, sess=None, output_tensor_name="output/BiasAdd:0", batch_size=64):
    """ run inference on new data """
    if sess is None and ckpt_prefix_fn is None:
        raise ValueError("must provide restored session or checkpoint from which to restore")

    created_sess = False
    if sess is None:
        sess = restore_sess(ckpt_prefix_fn)
        created_sess = True

    # get operations from the TensorFlow graph that are needed to run new examples through the network
    # placeholder for the input sequences
    input_seqs_ph = sess.graph.get_tensor_by_name("raw_seqs_placeholder:0")
    # predictions come from "output/BiasAdd:0" which refers to the final fully connected layer of the network
    # after the bias has been added. all my networks will have this same "output" layer. the extra squeeze
    # operation is added for data formatting
    predictions = sess.graph.get_tensor_by_name(output_tensor_name)

    # auto batch
    if encoded_data.shape[0] > batch_size:

        # run a single example through the network to get the shape of the output
        out_shape = list(np.squeeze(sess.run(predictions, feed_dict={input_seqs_ph: encoded_data[0:1, ...]})).shape)

        # set up an array to store outputs from each batch
        predictions_for_data = np.zeros([encoded_data.shape[0]] + out_shape)

        # compute the number of batches and split data into batches using the batch generator
        num_batches = int(np.ceil(encoded_data.shape[0]/batch_size))
        bg = batch_generator([encoded_data], batch_size, skip_last_batch=False, num_epochs=1, shuffle=False)

        for batch_num, batch_data in tqdm(enumerate(bg), total=num_batches):
            # if batch_num % 10 == 0:
            #     print("Running batch {} of {}...".format(batch_num+1, num_batches))
            batch_data = batch_data[0]
            predictions_for_batch = sess.run(predictions, feed_dict={input_seqs_ph: batch_data})
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            predictions_for_data[start_index:end_index, ...] = np.squeeze(predictions_for_batch)
    else:
        predictions_for_data = np.squeeze(sess.run(predictions, feed_dict={input_seqs_ph: encoded_data}))

    if created_sess:
        sess.close()

    return predictions_for_data


def main():
    pass


if __name__ == "__main__":
    main()
