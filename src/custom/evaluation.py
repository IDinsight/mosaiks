from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import r2_score


def show_results(y_test, y_pred, file_name=None, line=True, title=""):
    """
    Print stats and plot true vs predicted values.

    Parameters
    ----------
    y_test : array-like
        The true values
    y_pred : array-like
        The predicted values

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

    ax.set(xlabel="Predicted", ylabel="Observed")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.1, 1.1)

    if line:
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

    third_plot = True if y_pred_2_name!=None else False
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
    axes[1].set_title(y_pred_name) #"2011 Predicted"

    if third_plot:
        # predicted - scaled
        plot_colored_map(
            gdf,
            column=y_pred_2_name,
            vmin=vmin,
            vmax=vmax,
            ax=axes[2],
        )
        axes[2].set_title(y_pred_2_name) #"2011 Predicted (Scaled)"

    f.suptitle(title)
    plt.tight_layout()

    if file_name:
        file_path = (
            Path(__file__).parents[2] / "data" / "04_modeloutput" / f"{file_name}.png"
        )
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
