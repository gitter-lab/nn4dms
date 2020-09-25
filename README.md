# Neural networks for deep mutational scanning data
This repository is a supplement to our paper [Predicting Multi-Variant Protein Functional Activity From Deep Mutational Scanning Data](https://github.com/samgelman/nn4dms).
We trained and evaluated the performance of multiple types of neural networks on five deep mutational scanning datasets.
This repository contains code and examples that allow you to do the following:
 - Retrain the models from our publication
 - Train new models using our datasets or your own datasets
 - Use trained models to make predictions for new variants
 
## Setup
This code is based on Python 3.6 and TensorFlow 1.14. 
Use the provided [environment.yml](environment.yml) file to set up a suitable environment with [Anaconda](https://www.anaconda.com).

```
conda env create -f environment.yml
conda activate nn4dms
```

#### GPU support (optional)
By default, the environment uses CPU-only TensorFlow.
If you have an Nvidia GPU and want GPU support, use the [environment_gpu.yml](environment_gpu.yml) file instead. It will install `tensorflow-gpu` instead of `tensorflow`. You will need to make sure you have the [appropriate CUDA drivers and software](https://www.tensorflow.org/install/source#gpu) installed for TensorFlow 1.14: [cudnn 7.4](https://developer.nvidia.com/rdp/cudnn-archive) and [cudatoolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive). Certain versions of this Nvidia software may also be available for your operating system via Anaconda. 

#### Enrich2 (optional)
We used [Enrich2](https://github.com/FowlerLab/Enrich2) to compute functional scores for the GB1 and Bgl3 datsets. If you want to re-run that part of our pipeline, you must install Enrich2 according to the instructions on the Enrich2 GitHub page. Make sure the conda environment for Enrich2 is named "enrich2". This is optional; we provide pre-computed datasets in the [data](data) directory.

## Training a model
You can train a model by calling [code/regression.py](code/regression.py) with the required arguments specifying the dataset, network architecture, train-test split, etc.
For convenience, [regression.py](code/regression.py) accepts an arguments text file in addition to command line arguments. We provide a sample [arguments file](regression_args/example.txt) you can use as a template.

Call the following from the root directory to train a sample linear regression model on the avGFP dataset:
```
python code/regression.py @regression_args/example.txt 
```
The output, which includes the trained model, evaluation metrics, and predictions on each of the train/tune/tests sets, will automatically be placed in the [training_logs](output/training_logs) directory.

For a full list of parameters, call `python code/regression.py -h`.

#### Additional customization:
- To define your own custom network architecture, see the readme in the [network_specs](network_specs) directory.
- For more control over the train-test split, see the [train_test_split.ipynb](notebooks/train_test_split.ipynb) notebook.
- To compute your own protein structure graph for the graph convolutional network, see the [structure_graph.ipynb](notebooks/structure_graph.ipynb) notebook.
- To use your own dataset, see the readme in the [data](data) directory.

## Retraining models from our publication
In the [pub](pub) directory, we provide various files to facilitate retraining the models from our publication:
- The exact train/tune/test set splits
- Pre-made arguments files that can be fed directly into [regression.py](code/regression.py)

To retrain one of our models, call `python code/regression.py @pub/regression_args/<desired model>` from the root directory.

We also provide pre-trained models, similar to the ones from our publication, that can be used to predict scores for new variants. 

## Evaluating a model and making new predictions
During training, [regression.py](code/regression.py) saves a variety of useful information to the log directory, including predictions for all train, tune, and test set variants and the trained model itself. 

#### Evaluating a model
We provide convenience functions that allow you to easily load and process this log information. See the example in the [analysis.ipynb](notebooks/analysis.ipynb) notebook. You can also use TensorBoard to visualize how model performance changes during the training process. For more information, see the readme in the [training_logs](output/training_logs) directory.

#### Using a trained model to make predictions
For a straightforward example of how to use a trained model to make predictions, see the the [inference.ipynb](notebooks/inference.ipynb) notebook.



## External sources
[Our implementation](code/my_pipgcn.py) of graph convolutional networks is based on the implementation used in Protein Interface Prediction using Graph Convolutional Networks (https://github.com/fouticus/pipgcn). 

