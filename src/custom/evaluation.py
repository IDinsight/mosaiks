import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import r2_score


def show_results(y_test, y_pred):
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
    print("r2: %f" % r2_score(y_test, y_pred))
    print("pearson: %f" % pearsonr(y_test, y_pred)[0])
    print("kendall: %f" % kendalltau(y_test, y_pred)[0])
    print("spearman: %f" % spearmanr(y_test, y_pred)[0])

    # scatterplot
    ax = sns.scatterplot(x=y_pred, y=y_test)
    sns.lineplot(x=[0, 1], y=[0, 1], color="black", linestyle="--", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Observed")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.2, 1.1)
