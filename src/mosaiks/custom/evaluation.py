from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import precision_score, r2_score, recall_score


def get_cutoff_range_metrics(y_test, y_pred):
    """
    Run through a range of cutoff values and return the resulting performance metrics as a dataframe.

    Parameters
    ----------
    y_test : array-like
        The true values
    y_pred : array-like
        The predicted values

    Returns
    -------
    pd.DataFrame
        Table of metrics per cut-off value.

    """

    # recalls = []
    # precisions = []
    results = pd.DataFrame(
        columns=[
            "cutoff",
            "ground_pos",
            "pred_pos",
            "TP",
            "FP",
            "TN",
            "FN",
            "recall",
            "precision",
        ]
    )

    cutoffs = np.arange(0, 1.01, 0.01)
    for cutoff in cutoffs:
        y_test_binary = (y_test > cutoff).astype(int)
        y_pred_binary = (y_pred > cutoff).astype(int)

        # calculate total ground and predicted positives
        ground_pos = np.sum(y_test_binary)
        pred_pos = np.sum(y_pred_binary)

        # calculate and store true postives, false positives, etc.
        TP = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
        FP = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
        TN = np.sum((y_test_binary == 0) & (y_pred_binary == 0))
        FN = np.sum((y_test_binary == 1) & (y_pred_binary == 0))

        # calculate and store recall and precision
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)

        # append as row to results dataframe
        results.loc[cutoff] = [
            cutoff,
            int(ground_pos),
            pred_pos,
            TP,
            FP,
            TN,
            FN,
            recall,
            precision,
        ]

    # convert pandas types to ints
    results = results.astype(
        {
            "cutoff": "float",
            "ground_pos": "int",
            "pred_pos": "int",
            "TP": "int",
            "FP": "int",
            "TN": "int",
            "FN": "int",
            "recall": "float",
            "precision": "float",
        }
    )

    return results


def plot_metrics(results, experiment_name):
    """
    Create and save a 3x2 grid of plots showing precision, recall, and ROC.

    Parameters
    ----------
    results : pd.DataFrame
        Dataframe of metrics as created by `get_cutoff_range_metrics`.
    experiment_name: string
        Experiment name for title and filename,

    Returns
    -------
    None
    """
    cutoffs = results["cutoff"]
    recalls = results["recall"]
    precisions = results["precision"]

    # plots
    f, axes = plt.subplots(3, 2, figsize=(8, 12))

    # plot ground truth positive and predicted positive counts
    ax = axes[0, 0]
    ax.plot(cutoffs, results["ground_pos"], label="Ground Positives")
    ax.plot(cutoffs, results["pred_pos"], label="Predicted Positives")
    ax.set_title("No. true vs predicted positives")
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Count")
    ax.legend()

    # plot true and false positives/negatives
    ax = axes[0, 1]
    ax.plot(cutoffs, results["TP"], label="TP")
    ax.plot(cutoffs, results["FP"], label="FP")
    ax.plot(cutoffs, results["TN"], label="TN")
    ax.plot(cutoffs, results["FN"], label="FN")
    ax.set_title("No. of TP, FP, TN, FN")
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("")
    ax.legend()

    # plot recall and precision
    ax = axes[1, 0]
    ax.plot(cutoffs, recalls, label="Recall")
    ax.set_title("Recall per cutoff")
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Recall")

    ax = axes[1, 1]
    ax.plot(cutoffs, precisions, label="Precision")
    ax.set_title("Precision per cutoff")
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Precision")

    # plot precision-recall
    ax = axes[2, 0]
    ax.scatter(recalls, precisions, c=cutoffs, s=20, lw=0)
    ax.plot(recalls, precisions, label="Precision-Recall")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # plot ROC
    tpr = results["TP"] / (results["TP"] + results["FN"])
    fpr = results["FP"] / (results["FP"] + results["TN"])
    ax = axes[2, 1]
    cutoff_scatter = ax.scatter(fpr, tpr, c=cutoffs, s=20, lw=0)
    ax.plot(fpr, tpr, label="ROC")
    ax.set_title("Bonus: ROC")
    ax.set_xlabel("FP Rate")
    ax.set_ylabel("TN Rate")
    plt.colorbar(cutoff_scatter, ax=ax, label="Cutoff")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).parents[2]
        / "data"
        / "04_modeloutput"
        / f"precision_recall_{experiment_name}.png",
        dpi=300,
    )
    plt.show()


def show_results(
    y_test,
    y_pred,
    xlabel="Predicted 2013",
    ylabel="Observed",
    file_name=None,
    line=True,
    title="",
):
    """
    Print stats and plot true vs predicted values.

    Parameters
    ----------
    y_test : array-like
        The true values
    y_pred : array-like
        The predicted values
    xlabel : str
        The label for the x-axis
    ylabel : str
        The label for the y-axis
    file_name : str
        The name of the file to save the plot to.
    line : bool
        Whether to plot a line at y=x.
    title : str
        The title of the plot.

    Returns
    -------
    None

    """

    # calculate the stats
    r2 = r2_score(y_test, y_pred).round(3)
    pearson = pearsonr(y_test, y_pred)[0].round(3)
    spearman = spearmanr(y_test, y_pred)[0].round(3)
    kendall = kendalltau(y_test, y_pred)[0].round(3)

    stats_text = f"""R_squared  {r2}
Pearson       {pearson}
Spearman   {spearman}
Kendall        {kendall}"""

    # scatterplot
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(
        x=y_pred,
        y=y_test,
        ax=ax,
        alpha=0.6,
        s=15,
        linewidth=0.3,
    )

    # add the stats
    ax.text(
        0.05,
        0.8,
        stats_text,
        bbox=dict(
            facecolor="white",
            edgecolor="grey",
            alpha=0.5,
        ),
        transform=ax.transAxes,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    if line:
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.1, 1.1)
        sns.lineplot(
            x=[0, 1],
            y=[0, 1],
            color="black",
            linestyle="--",
            alpha=0.6,
            ax=ax,
        )

    if file_name:
        file_path = (
            Path(__file__).parents[2] / "data" / "04_modeloutput" / f"{file_name}.png"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")


def plot_colored_map(gdf, column, vmin, vmax, ax):
    gdf.plot(
        column=column,
        # alpha=0.5,
        markersize=0.1,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.axis("off")


def plot_prediction_maps(
    gdf,
    y_name,
    y_pred_name="predicted",
    y_pred_2_name=None,
    vmin=0,
    vmax=1,
    file_name=None,
    title="",
):
    """
    Plot the true and predicted values on a map.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the true and predicted values.
    y_name : str
        The name of the column containing the true values.
    y_pred_name : str
        The name of the column containing the predicted values.
    y_pred_2_name : str
        The name of the column containing other predicted values to plot in a 3rd plot.
    vmin, vmax : float
        The minimum and maximum value for the colorbar.
    file_name : str
        The name of the file to save the plot to.

    Returns
    -------
    None

    """

    third_plot = True if y_pred_2_name != None else False
    f, axes = plt.subplots(1, 2 + third_plot, sharey=True, figsize=(10, 5))

    # observed
    plot_colored_map(
        gdf,
        column=y_name,
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
    )
    axes[0].set_title("2011 Observed")

    # predicted
    plot_colored_map(
        gdf,
        column=y_pred_name,
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
    )
    axes[1].set_title(y_pred_name)  # "2011 Predicted"

    if third_plot:
        # predicted - scaled
        plot_colored_map(
            gdf,
            column=y_pred_2_name,
            vmin=vmin,
            vmax=vmax,
            ax=axes[2],
        )
        axes[2].set_title(y_pred_2_name)  # "2011 Predicted (Scaled)"

    f.suptitle(title)
    plt.tight_layout()

    if file_name:
        file_path = (
            Path(__file__).parents[2] / "data" / "04_modeloutput" / f"{file_name}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
