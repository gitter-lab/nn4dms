# Network architecture specifications

This directory contains neural network architectures defined in json files.
This is a rudimentary format I created to help with reproducibility and parameter sweeps. It is not an official TensorFlow format, and thus it is quite limited. It supports the following layers: dense, sequence convolutional, graph convolutional, dropout, and flatten. For activation functions, it supports ReLU, leaky ReLU, and "none". You can explore the network definitions in this directory to get an idea of how it works. You should be able to easily change some parameters the number of layers, filters, and hidden nodes.

## Creating your own network architecture
The network architecture json files are parsed in [build_tf_model.py](../code/build_tf_model.py) using the function `bg_inference()`. You can follow along with the code to figure out how to add support for additional types of layers or parameters. 

Alternatively, you can abandon the json format and define your models programmatically, as they do in the TensorFlow documentation using the `tf.layers` api. This would work with the overall framework as long as `bg_inference()` returns a tensor containing the predicted scores. Your code would need to take the raw sequences input, contained in `ph_inputs_dict["raw_seqs"]`, and add operations to build the "inference" part of the graph that goes from raw sequences to predicted scores. 