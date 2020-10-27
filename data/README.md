# Data

Each dataset has its own directory containing the following:
- The dataset .tsv file, a tab-separated dataframe with columns for the variant and functional score. 
- A pdb file defining the 3D structure of the protein, used for generating protein structure graphs for graph convolutional networks. 
- The graphs directory containing protein structure graphs. Some graphs have been pre-computed and provided here for reference. You can also generate your own protein structure graphs with different thresholds. See the example notebook [structure_graph.ipynb](../notebooks/structure_graph.ipynb). 
- A splits directory containing train-tune-test splits that have been saved as files for easy reuse and reproducibility. No splits are provided, so the splits directory does not show up in GitHub, but it will be generated when you generate your first train-tune-test split. See the notebook [train_test_split.ipynb](../notebooks/train_test_split.ipynb) for a guide on how to generate a train-tune-test split. 

There is also a [directory](aaindex) for the processed AAindex features. These were generated with [parse_aaindex.py](../code/parse_aaindex.py) and are used for encoding variants.
See the [source_data](../source_data) directory for references to the original datasets.

### Using your own dataset
We recommend following our format if you want to use your own dataset with this codebase. Create a folder for your dataset and add the corresponding files as described above (see the current datasets for reference). Although not required, we also recommend you define your dataset in [constants.py](../code/constants.py). Doing so will make things easier because you will be able to simply specify the dataset name in several places where otherwise you'd need to specify paths, files, etc. 
