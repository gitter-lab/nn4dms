""" Compute evaluation metrics for regression """

import sklearn.metrics as skm
import scipy
import scipy.stats


def compute_metrics(true_scores, predicted_scores, metrics=("mse", "pearsonr", "r2", "spearmanr")):

    metrics_dict = {}
    # add scores to evaluation dict
    metrics_dict["true"] = true_scores
    metrics_dict["predicted"] = predicted_scores

    # compute requested metrics
    for metric in metrics:
        if metric == "mse":
            metrics_dict["mse"] = skm.mean_squared_error(true_scores, predicted_scores)
        elif metric == "pearsonr":
            metrics_dict["pearsonr"] = scipy.stats.pearsonr(true_scores, predicted_scores)[0]
        elif metric == "spearmanr":
            metrics_dict["spearmanr"] = scipy.stats.spearmanr(true_scores, predicted_scores)[0]
        elif metric == "r2":
            metrics_dict["r2"] = skm.r2_score(true_scores, predicted_scores)

    return metrics_dict


def main():
    pass


if __name__ == "__main__":
    main()
