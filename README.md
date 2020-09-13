# Neural networks for deep mutational scanning data
This repository is a supplement to our paper [Predicting Multi-Variant Protein Functional Activity From Deep Mutational Scanning Data](https://github.com/samgelman/nn4dms).
We trained and evaluated the performance of multiple types of neural networks on five deep mutational scanning datasets.
This repository contains code and examples that allow you to do the following:
 - Re-run the pipeline we used to train models for our publication
 - Train new models using your own datasets
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
If you have an Nvidia GPU and want GPU support, use the [environment_gpu.yml](environment_gpu.yml) file instead. It will install `tensorflow-gpu` instead of `tensorflow`. You will need to make sure you have the [appropriate CUDA drivers and software](https://www.tensorflow.org/install/source) installed for TensorFlow 1.14.

#### Enrich2 (optional)
We used [Enrich2](https://github.com/FowlerLab/Enrich2) to compute functional scores for the GB1 and Bgl3 datsets. If you want to re-run that part of our pipeline, you must install Enrich2 according to the instructions on the Enrich2 GitHub page. Make sure the conda environment for Enrich2 is named "enrich2". This is optional; we provide pre-computed datasets in the [data](data) directory.

## Quickstart 
Quickstart guide goes here. Explanations for the different files, directories, etc. 

## External sources
[Our implementation](code/my_pipgcn.py) of graph convolutional networks is based on the implementation used in Protein Interface Prediction using Graph Convolutional Networks (https://github.com/fouticus/pipgcn). 
