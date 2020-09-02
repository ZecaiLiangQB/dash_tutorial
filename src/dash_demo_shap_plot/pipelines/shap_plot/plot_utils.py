# QUANTUMBLACK CONFIDENTIAL
#
# Copyright (c) 2016 - present QuantumBlack Visual Analytics Ltd. All
# Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# QuantumBlack Visual Analytics Ltd. and its suppliers, if any. The
# intellectual and technical concepts contained herein are proprietary to
# QuantumBlack Visual Analytics Ltd. and its suppliers and may be covered
# by UK and Foreign Patents, patents in process, and are protected by trade
# secret or copyright law. Dissemination of this information or
# reproduction of this material is strictly forbidden unless prior written
# permission is obtained from QuantumBlack Visual Analytics Ltd.

"""
Run SHAP and generalte all the reporting files.

Functions:
    "plot_dependence_with_histogram",
    "plot_dependence_with_histogram_color_by_segment",
    "_calculate_contri_df",
    "_plot_median",
    "_set_axis_limit",
    "_plot_histogram",

"""

import pandas as pd
import numpy as np
import shap
import matplotlib
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def plot_dependence_with_histogram(
    col,
    shap_value,
    X,
    holdout_data,
    single_variable=False,
    variable_sum=False,
    interaction_col="auto",
    plt_class_col=None,
    filtered_index=None,
    show_median=False,
    selected_xlim=None,
    selected_ylim=None,
    population_histogram=True,
    target_label=None,
    nbins=35,
):
    """Plot dependence plot with overlapping histogram and twin y-axes.

    Args:
        col (string): Name of the column that we want to visualize
        shap_value (numpy.array): The shap value matrix given by explainer.shap_values(X)[1]
        X (pandas.DataFrame): The test data with selected features
        holdout_data (pandas.DataFrame): The test data with all columns
        single_variable (boolean): Whether to consider interaction terms
        variable_sum (boolean): Whether to add shap values of variable col and interactive variable
        interaction_col (None or string): The column name for plot class.
        plt_class_col (None or string): The column name for interaction terms.
                                        'auto' means asking SHAP to select the one interacting most.
        filtered_index (None or list): If not None, it should be a list of Boolean values indicating
                                        whether this indice should be in the visualization
                                       If None, not filtering is done on the holdout data.
        show_median (boolean): Whether to show the median line for the interaction feature
        selected_xlim (list of two int or None): Lower bound and upper bound of the x axis
        selected_ylim (list of two int or None): Lower bound and upper bound of the y axis
        population_histogram (boolean): If True, a population histogram will be ploted in the background
        target_label (string): The name of the target shows on the plot
        nbins (int): The number of bins in histogram

    Returns:
        Prints out average probability of initiation before and after applying action
        for that particular cohort.
        Creates histogram of predictive probabilities before and after action

    """
    shap_values = shap_value.copy()
    X = X.copy()
    holdout_data = holdout_data.copy()

    # TODO: not really functioning code, pass the filtered data as input instead
    if filtered_index is not None:
        shap_values = shap_values[filtered_index, :]
        X = X.loc[filtered_index, :]
        holdout_data = holdout_data.loc[filtered_index, :]

    # change default parameters for binary variable
    # TODO: this counts df is never used anywhere
    if X[col].nunique() <= 3:
        counts = pd.DataFrame(X[col].value_counts().reset_index())
        counts.columns = [col, "Count"]
        # print(counts.to_string(index=False), "\n\n\n")

    # replace the shap value by the sum of the shap value
    if variable_sum:
        shap_sum = (
            shap_values[:, np.where(X.columns == col)[0][0]]
            + shap_values[:, np.where(X.columns == interaction_col)[0][0]]
        )
        shap_values[:, np.where(X.columns == col)[0][0]] = shap_sum

    # TODO: when plt_class_col is continuous
    # choose category for plotting
    if plt_class_col is None and interaction_col != "auto":
        plt_class_col = interaction_col

    # build contri_df for median lines
    contri_df_agg = _calculate_contri_df(
        col, X, shap_values, holdout_data, plt_class_col
    )

    # SHAP plot
    shap_args = {
        "ind": col,
        "shap_values": shap_values,
        "features": X,
        "interaction_index": interaction_col,
        "alpha": 0.5,
        "dot_size": 10,
        "show": False,
    }
    # single variable
    if single_variable:
        shap_args["color"] = "purple"
        shap.dependence_plot(**shap_args)
        plt.title("{} Contribution to {}".format(col, target_label))

    # use the interaction_col for interaction
    else:
        shap.dependence_plot(**shap_args)
        plt.title("{} Contribution to {}".format(col, target_label))
        if variable_sum:
            plt.ylabel("Sum of SHAP Value")

    # plot median
    _plot_median(show_median, contri_df_agg, col, plt_class_col)

    # set axis limit
    _set_axis_limit(selected_xlim, selected_ylim)

    # plot historgram when the feature is not a binary variable
    _plot_histogram(population_histogram, col, holdout_data, nbins)

    plt.gcf().set_size_inches(15, 10)
    # plt.show()


def plot_dependence_with_histogram_color_by_segment(
    col,
    shap_value,
    X,
    holdout_data,
    plt_class_col=None,
    filtered_index=None,
    show_median=False,
    selected_xlim=None,
    selected_ylim=None,
    population_histogram=True,
    target_label=None,
    nbins=35,
):
    """Plot dependence plot with overlapping histogram.

    Color-code the dots and median line by segment flags (may not be in selected features).

    Args:
        col (string): Name of the column that we want to visualize
        shap_value (numpy.array): The shap value matrix given by explainer.shap_values(X)[1]
        X (pandas.DataFrame): The test data with selected features
        holdout_data (pandas.DataFrame): The test data with all columns
        plt_class_col (None or string): The column name as segment flag. Can be any column in holdout_data.
        filtered_index (None or list): If not None, it should be a list of Boolean values indicating
                                        whether this indice should be in the visualization
                                       If None, not filtering is done on the holdout data.
        show_median (boolean): Whether to show the median line for the interaction feature
        selected_xlim (list of two int or None): Lower bound and upper bound of the x axis
        selected_ylim (list of two int or None): Lower bound and upper bound of the y axis
        population_histogram (boolean): If True, a population histogram will be ploted in the background
        target_label (string): The name of the target shows on the plot
        nbins (int): The number of bins in histogram

    Returns:
        Scatter plot of SHAP values for the selected feature.
        Creates histogram of the selected feature in the background.
        Color-code the scatter plot by selected flag.
        Color-code the median line by selected flag when show_median = True.

    """
    shap_values = shap_value.copy()
    X = X.copy()
    holdout_data = holdout_data.copy()

    # TODO: not really functioning code, pass the filtered data as input instead
    if filtered_index is not None:
        shap_values = shap_values[filtered_index, :]
        X = X.loc[filtered_index, :]
        holdout_data = holdout_data.loc[filtered_index, :]

    # build contri_df for median lines
    contri_df_agg = _calculate_contri_df(
        col, X, shap_values.values, holdout_data, plt_class_col
    )

    # TODO: when plt_class_col is continuous
    # create a list of color based on values of column plt_class_col
    cmap = matplotlib.cm.get_cmap("Set1")
    color_dir = {}
    for i, cat in enumerate(holdout_data[plt_class_col].unique()):
        color_dir[cat] = cmap.colors[i]
    color_col = [color_dir[value] for value in holdout_data[plt_class_col]]

    # scatter plot
    plt.scatter(X[col], shap_value[col], c=color_col, s=10, alpha=0.5)
    # plot title
    plt.title("{} Contribution to {}".format(col, target_label), fontsize=15)
    plt.xlabel("Feature: " + col, fontsize=15)
    plt.ylabel("SHAP value", fontsize=15)
    # create legend
    legend_elements = []
    for value, color in color_dir.items():
        legend_elements.append(
            Patch(facecolor=color, edgecolor=color, label=plt_class_col + ": " + value)
        )
    plt.legend(handles=legend_elements)

    # plot median
    _plot_median(show_median, contri_df_agg, col, plt_class_col, color_dir=color_dir)

    # set axis limit
    _set_axis_limit(selected_xlim, selected_ylim)

    # plot historgram when the feature is not a binary variable
    _plot_histogram(population_histogram, col, holdout_data, nbins)

    plt.gcf().set_size_inches(15, 10)
    # plt.show()


def _calculate_contri_df(col, X, shap_values, holdout_data, plt_class_col):
    """Calculate the median SHAP values for each distinct feature value.

    If segment flag (plt_class_col) is assigned, calculate by group separately.

    Args:
        col (string): Name of the column that we want to visualize
        X (pandas.DataFrame): The test data with selected features
        shap_values (numpy.array): The shap value matrix given by explainer.shap_values(X)[1]
        holdout_data (pandas.DataFrame): The test data with all columns
        plt_class_col (string): Name of the column we use to color-code scatter plot.
                                Doesn't need to be a feature in the model.

    Returns:
        If group flag (plt_class_col) is assigned:
            a pandas.DataFrame with columns [feature-name, flag-name, median-SHAP-value]
        Else:
            a pandas.DataFrame with columns [feature-name, median-SHAP-value]

    """
    if plt_class_col is not None:
        contri_df = pd.DataFrame.from_dict(
            {
                col: X[col],
                plt_class_col: holdout_data[plt_class_col],
                "shap_value": shap_values[:, np.where(X.columns == col)[0][0]],
            }
        )
        contri_df_agg = contri_df.groupby([col, plt_class_col]).median().reset_index()
        contri_df_agg.columns = [col, plt_class_col, "Median"]

    else:
        contri_df = pd.DataFrame.from_dict(
            {
                col: X[col],
                "shap_value": shap_values[:, np.where(X.columns == col)[0][0]],
            }
        )
        contri_df_agg = contri_df.groupby([col]).median().reset_index()
        contri_df_agg.columns = [col, "Median"]

    return contri_df_agg


def _plot_median(show_median, contri_df_agg, col, plt_class_col, color_dir=None):
    """Plot the median SHAP values on top of scatter plot.

    Plot each group separately if segment flag (plt_class_col) is assigned.

    Args:
        show_median (boolean): Whether to show the median line for the interaction feature
        contri_df_agg (pandas.DataFrame): df with columns [feature-name, flag-name, median-SHAP-value]
                                         if plt_class_col not None,
                                         else df with columns [feature-name, median-SHAP-value].
        col (string): Name of the column that we want to visualize
        plt_class_col (string): Name of the column we use to color-code scatter plot.
                                Doesn't need to be a feature in the model.
        color_dir (dict): a dictionary that maps distinct values of the flag feature to colors,
                         only use when plt_class_col is not None.

    Returns:
        The median line plot with legend.
        Color-coded by segment flag (plt_class_col) if it's assigned.

    """
    if show_median:
        if plt_class_col is None:
            plt.plot(
                col,
                "Median",
                data=contri_df_agg,
                label="Median Contribution for {}".format(col),
            )
        else:
            # re-create color dictionary when plt_class_col is assigned,
            # but no color_dir is passed
            if color_dir is None:
                # create a list of color based on values of column plt_class_col
                cmap = matplotlib.cm.get_cmap("Set1")
                color_dir = {}
                for i, cat in enumerate(contri_df_agg[plt_class_col].unique()):
                    color_dir[cat] = cmap.colors[i]

            for cur_cate in contri_df_agg[plt_class_col].unique():
                plt.plot(
                    col,
                    "Median",
                    data=contri_df_agg.loc[contri_df_agg[plt_class_col] == cur_cate, :],
                    label="Median Contribution for "
                    + plt_class_col
                    + "=={}".format(cur_cate),
                    color=color_dir[cur_cate],
                )
        # print(contri_df_agg)
        plt.legend()


def _set_axis_limit(selected_xlim, selected_ylim):
    """Manually set the axis limits for x-axis and y-axis.

    Args:
        selected_xlim (list of two ints or None): lower bound and upper bound of the x axis
        selected_ylim (list of two ints or None): lower bound and upper bound of the y axis

    Returns:
        Change the axis limits of the plot inplace.

    """
    if selected_xlim is not None:
        cur_axis = plt.gca()
        cur_axis.set_xlim(selected_xlim)

    if selected_ylim is not None:
        cur_axis = plt.gca()
        cur_axis.set_ylim(selected_ylim)


def _plot_histogram(population_histogram, col, holdout_data, nbins):
    """Plot histogram of the selected feature in the background.

    Args:
        population_histogram (boolean): If True, a population histogram will be plotted in the background
        col (string): Name of the column that we want to visualize
        holdout_data (pandas.DataFrame): The test data with all columns
        nbins (int): The number of bins in histogram

    Returns:
        Histogram plot in the background.

    """
    if population_histogram:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.hist(
            holdout_data[col],
            bins=nbins,
            alpha=0.3,
            label="Population Count",
            color="grey",
        )
        plt.legend(loc="upper left")
