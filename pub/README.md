# Publication directory
We provide various files to facilitate retraining the models from the publication.

The [splits](splits) directory contains the exact train/tune/test sets we used for each dataset. 
These splits consist of numerical indices for each of the sets that index into the corresponding dataset .tsv files in the [data](../data) directory.
See [train_test_split.ipynb](../notebooks/train_test_split.ipynb) for information about the standard splits and [extrapolation.ipynb](../notebooks/extrapolation.ipynb) for information about the positional and mutational based splits.

The [regression_args](regression_args) directory contains argument files meant to be used with [regression.py](../code/regression.py). 
These argument files have been set to use the same train/test splits, network architectures, and hyperparameters we used to train the models in the publication. 
Simply call `python code/regression.py @pub/regression_args/<desired model>` from the root directory to train your desired model.
The output, which includes the trained model, evaluation metrics, and predictions on each of the train/tune/tests sets, will automatically be placed in [training_logs](../output/training_logs).
Due to the stochastic nature of training neural networks, your results may not match ours exactly, but they should be fairly close.
  
The [trained_models](trained_models) directory contains pre-trained models that are similar to the ones from the publication. 
You can use these to perform inference and make predictions for new variants by following the example in the [inference.ipynb](../notebooks/inference.ipynb) notebook.
These models are retrainings of the models used in the publication, and again, due to the stochastic nature of training neural networks, they may not match the models from the publication exactly. 