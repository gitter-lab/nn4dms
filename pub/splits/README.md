# Train-tune-test splits

See the methods section of the manuscript for more detailed descriptions and how these splits were used to test the models' performance and generalization.

| Split | Description |
|---------|---------------------------------------------------------------------------------------------------|
| main   | The main split using the full dataset size                                         |
| main_replicates    | Replicates of the main split with different random seeds                       |
| reduced     | Splits with reduced training set sizes |
| mutation_extrapolation    | Splits where the test set contains no informational overlap with the train and tune sets in terms of specific mutations  |
| position_extrapolation   | Splits where the test set contains no informational overlap with the train and tune sets in terms of positions where mutations occur  |
