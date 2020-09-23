# Network architecture specifications

This directory contains neural network architectures defined in yaml files.
This is a simple format I created to help with reproducibility and parameter sweeps. 
It is not an official TensorFlow format, and thus it may be limited. 

A network yaml file contains a list of layers that compose the network. 
Each layer consists of a main function, such as `tf.layers.dense` for a fully connected layer.
Each layer also has a list of keyword arguments, such as `name`, `units`, and `activation`, that correspond to parameters of the main function. 

Here is a sample network definition that contains three layers: flatten, dense, and dropout. 
```
network:
  - layer_func: ~tf.layers.flatten
    arguments:
      inputs: ~layers[-1]

  - layer_func: ~tf.layers.dense
    arguments:
      name: dense1
      inputs: ~layers[-1]
      units: 1000
      activation: ~tf.nn.leaky_relu

  - layer_func: ~tf.layers.dropout
    arguments:
      inputs: ~layers[-1]
      rate: 0.2
      training: ~ph_inputs_dict["training"]
```

In order for a function or variable defined in the yaml file to be parsed into python code, you must prefix it with `~`. This tells the parser to call `eval()` on the subsequent text.

You can explore the network definitions in this directory to get an idea of how this format works for other layers. You should be able to easily change parameters such as the number of layers, filters, and hidden nodes.

## Creating your own network architecture

The network architecture yaml files are parsed in [build_tf_model.py](../code/build_tf_model.py) using the function `bg_inference()`. 

There are a few key variables that you can use in the yaml file. 
- `~layers` is an array containing the outputs from previous layers, so `~layers[-1]` refers to the output from the preceding layer.  
- `~ph_inputs_dict["training"]` is a boolean variable that is set to True if the network is training and false if the network is performing inference.
- `~adj_mtx` is the adjacency matrix for the protein structure graph, if specified.

In general, as long as you prefix the call with `~`, you can use any function or variable in global or local scope in the `bg_inference()` function.

This format should be able to support most layer functions in the `tf.layers` api. You can also define your own custom layers, like we did for graph convolutions using `node_average_gc()` in [my_pipgcn.py](../code/my_pipgcn.py). 

We recommend looking at the examples in this directory to get a better sense of how to define a network architecture. 

##### Defining a network programmatically 
Alternatively, you can abandon the yaml format and define your models programmatically, as they do in the TensorFlow documentation using the `tf.layers` api. This would work with the overall framework as long as `bg_inference()` returns a tensor containing the predicted scores. Your code would need to take the raw sequences input, contained in `ph_inputs_dict["raw_seqs"]`, and add operations to build the "inference" part of the graph that goes from raw sequences to predicted scores. 