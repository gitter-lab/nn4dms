# Arguments for regression.py

This directory is a convenient place to put arguments files for [regression.py](../code/regression.py).
These argument files contain all the arguments needed to train a model. 
Simply call `python code/regression.py @regression_args/<argument file>` from the root directory to train a model using your desired argument file. We have provided a sample file, [example.txt](example.txt), that you can use as a template. It does not contain every possible argument. Call `python code/regression.py -h` for a printout with a full listing of possible arguments. 

### Generating arguments for a hyperparameter sweep
This codebase contains builtin functionality for generating argument files for a hyperparameter sweep. The argument files are generated from a template file where you can specify multiple possible values for each parameter. We provided a sample template file: [example_run.yml](example_run.yml). This file can be used with [gen_args.py](../code/gen_args.py). Call `python code/gen_args.py -f regression_args/example_run.yml`. The script will automatically generate argument files corresponding to all possible combinations of arguments given in the template file. Those argument files can then be used with [regression.py](../code/regression.py).