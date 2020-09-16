# Training logs
This is the default directory where [regression.py](../../code/regression.py) will save trained models, evaluations, etc. 
You can specify a different directory for training logs with the `--log_dir_base` argument.
If you're training multiple models that are logically grouped together, like for a specific dataset, it may make sense to save those logs to a different, group-specific directory where they can be easily processed as a group. 
For an example of how to load and evaluate these logs, see the [analysis.ipynb](../../notebooks/analysis.ipynb) notebook.


The training logs saved by [regression.py](../../code/regression.py) are compatible with TensorBoard, a visualization tool built into TensorFlow.
You can start TensorBoard by calling `tensorboard --logdir=<log directory for model>`. Then, navigate to the URL displayed in the terminal to access TensorBoard. 
The script is set up save mean squared error, Pearson's and Spearman's correlation coefficients, and r^2 values at regular intervals during training.
This lets you easily observe how performance of the model changes during training.
You can specify how often you want these metrics to be saved using the `--epoch_evaluation_interval` argument to [regression.py](../../code/regression.py).


Finally, if you're using early stopping, [regression.py](../../code/regression.py) will save two trained models: the lowest loss model and the final model. They are identified by the epoch number in the filename. The lowest loss model will have a lower epoch number, and that is the one you'll likely want to use. For an example of how to use a trained model to perform inference and make predictions for new variants, see the [inference.ipynb](../../notebooks/inference.ipynb) notebook. 