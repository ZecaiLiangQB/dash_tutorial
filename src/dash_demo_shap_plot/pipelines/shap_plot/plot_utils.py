# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for post model SHAP analysis in notebook.

Used by /home/98_examples/template_plot_SHAP_results.ipynb,
to plot partial dependent plot,
and to plot partial dependent plot by segments.
"""

from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.patches import Patch


def plot_shap_dependence_plot_with_interaction(
    feature_col: str,
    shap_value_df: pd.DataFrame,
    data_df: pd.DataFrame,
    interaction_col: Union[str, None] = None,
    plot_median_line: bool = False,
    selected_xlim: Union[List[float], None] = None,
    selected_ylim: Union[List[float], None] = None,
    figsize: List[int] = [15, 10],
    plot_histogram: bool = True,
    nbins: int = 35,
) -> None:
    """Plots dependence plot with overlapping histogram and interaction column.

    Args:
        feature_col (str): name of the column that we want to visualize
        shap_value_df (pd.DataFrame): the shap value matrix given by explainer.shap_value_df(data_df)
        data_df (pd.DataFrame): the test data with selected features
        interaction_col (None or str): the column name for interaction term.
                                       'auto' means asking SHAP to select interaction term,
                                       None means no interaction term.
        plot_median_line (boolean): whether to show the median line for the interaction feature
        selected_xlim (list of two float or None): lower bound and upper bound of the x axis
        selected_ylim (list of two float or None): lower bound and upper bound of the y axis
        figsize (list of two int): size of the plot
        plot_histogram (boolean): if True, a population histogram will be plotted in the background
        nbins (int): the number of bins in histogram

    Returns:
        Scatter plot of SHAP values for the selected feature.
        Creates histogram of the selected feature in the background.
        Interaction column on the right-side panel.
        Color-code the scatter plot by interaction column.

    """
    # Retrieve configuration parameters
    selected_feature_cols = shap_value_df.columns

    # Build contri_df for median lines
    median_shap_df = _calculate_median_shap_df(feature_col, data_df, shap_value_df)

    # SHAP plot
    shap_args = {
        "ind": feature_col,
        "shap_values": shap_value_df.loc[:, selected_feature_cols].values,
        "features": data_df.loc[:, selected_feature_cols],
        "interaction_index": interaction_col,
        "alpha": 0.5,
        "dot_size": 10,
        "show": False,
    }
    shap.dependence_plot(**shap_args)
    plt.title("{} Contribution to {}".format(feature_col, "target variable"))

    # Plot median SHAP line
    _plot_median(plot_median_line, median_shap_df, feature_col, color_dir=None)

    # Set axis limit
    _set_axis_limit(selected_xlim, selected_ylim)

    # Plot histogram when the feature is not a binary variable
    _plot_histogram(plot_histogram, feature_col, data_df, nbins)

    # plt.gcf()
    # plt.show()


def plot_shap_dependence_plot_by_segment(
    params: dict,
    feature_col: str,
    shap_value_df: pd.DataFrame,
    data_df: pd.DataFrame,
    segment_col: Union[str, None] = None,
    plot_median_line: bool = False,
    selected_xlim: Union[List[float], None] = None,
    selected_ylim: Union[List[float], None] = None,
    figsize: List[int] = [15, 10],
    plot_histogram: bool = True,
    nbins: int = 35,
) -> None:
    """Plots dependence plot with overlapping histogram and by segment.
    Color-code the dots and median line by segment flags (may not be in selected features).

    Args:
        params (dict): global parameters from parameters.yml
        feature_col (str): name of the column that we want to visualize
        shap_value_df (pd.DataFrame): the shap value matrix given by explainer.shap_value_df(data_df)
        data_df (pd.DataFrame): the test data with selected features.
        segment_col (None or str): the column name as segment group.
        plot_median_line (boolean): whether to show the median line for shap values.
        selected_xlim (list of two float or None): lower bound and upper bound of the x axis
        selected_ylim (list of two float or None): lower bound and upper bound of the y axis
        figsize (list of two int): size of the plot.
        plot_histogram (boolean): if True, a population histogram will be plotted in the background.
        nbins (int): the number of bins in histogram.

    Returns:
        Scatter plot of SHAP values for the selected feature.
        Creates histogram of the selected feature in the background.
        Color-code the scatter plot by selected flag.
        Color-code the median line by selected flag when plot_median_line = True.

    """
    # Retrieve configuration parameters
    numeric_cols = params["numeric_cols"]
    target_variable_col = params["target_var"]

    # Build contri_df for median lines
    median_shap_df = _calculate_median_shap_df(
        feature_col, data_df, shap_value_df, segment_col
    )

    # Color by segment_col
    if segment_col is not None:
        if segment_col not in numeric_cols:
            color_col, color_dir = _color_by_segment_col(data_df[segment_col])
        else:
            # Bin continuously feature into quartiles
            segment_col_cat = pd.qcut(data_df[segment_col], 4)
            color_col, color_dir = _color_by_segment_col(segment_col_cat)
    else:
        color_col = None
        color_dir = None

    # Plot scatter plot
    plt.scatter(
        data_df[feature_col], shap_value_df[feature_col], c=color_col, s=10, alpha=0.5
    )

    # Plot title
    plt.title(
        "{} Contribution to {}".format(feature_col, target_variable_col), fontsize=15
    )
    plt.xlabel("Feature: " + feature_col, fontsize=15)
    plt.ylabel("SHAP value", fontsize=15)

    # Create legend
    if segment_col is not None:
        legend_elements = []
        for value, color in color_dir.items():
            legend_elements.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    label=segment_col + ": " + str(value),
                )
            )
        plt.legend(handles=legend_elements)

    # Plot median SHAP line
    _plot_median(
        plot_median_line, median_shap_df, feature_col, segment_col, color_dir=color_dir
    )

    # Set axis limit
    _set_axis_limit(selected_xlim, selected_ylim)

    # Plot histogram when the feature is not a binary variable
    _plot_histogram(plot_histogram, feature_col, data_df, nbins)

    plt.gcf().set_size_inches(figsize[0], figsize[1])
    # plt.show()


def _calculate_median_shap_df(
    feature_col: str,
    data_df: pd.DataFrame,
    shap_value_df: pd.DataFrame,
    segment_col: str = None,
) -> pd.DataFrame:
    """Calculates the median SHAP values for each distinct feature value.
    If segment group (segment_col) is assigned, calculate by group separately.

    Args:
        feature_col (str): name of the column that we want to visualize
        data_df (pd.DataFrame): the test data with selected features
        shap_value_df (pd.DataFrame): the shap value matrix given by explainer.shap_values(data_df)
        segment_col (str): name of the column we use to color-code scatter plot.
                           Doesn't need to be a feature in the model.

    Returns:
        if segment column (segment_col) is assigned:
            a pandas.DataFrame with columns [feature-name, segment-name, median-SHAP-value]
        else:
            a pandas.DataFrame with columns [feature-name, median-SHAP-value]

    """
    if segment_col is not None:
        # Assemble data for median line
        contri_df = pd.DataFrame.from_dict(
            {
                feature_col: data_df[feature_col].values,
                segment_col: data_df[segment_col].values,
                "shap_value_df": shap_value_df[feature_col].values,
            }
        )
        # Aggregate by segment column, and feature column
        contri_df_agg = (
            contri_df.groupby([feature_col, segment_col]).median().reset_index()
        )
        contri_df_agg.columns = [feature_col, segment_col, "Median"]

    else:
        # Assemble data for median line
        contri_df = pd.DataFrame.from_dict(
            {
                feature_col: data_df[feature_col].values,
                "shap_value_df": shap_value_df[feature_col].values,
            }
        )
        # Aggregate by feature column
        contri_df_agg = contri_df.groupby([feature_col]).median().reset_index()
        contri_df_agg.columns = [feature_col, "Median"]

    return contri_df_agg


def _color_by_segment_col(segment_col_df: pd.DataFrame) -> Tuple[list, dict]:
    """Generates the value for color column based on value of the input column.

    Args:
        segment_col_df (pd.Series): the column to segment and color by data points.
                                    If continuous, bin values into quartiles.

    Returns:
        color_col (list): assign color to each data points based on the value of segment_col.
        color_dir (dict): color dict that map each segment to its color.

    """
    # Choose cmap for colors
    cmap = matplotlib.cm.get_cmap("Set1")
    color_dir = {}

    # Create cmap for each segment
    for i, cat in enumerate(segment_col_df.unique()):
        color_dir[cat] = cmap.colors[i]
    color_col = [color_dir[value] for value in segment_col_df]

    return (color_col, color_dir)


def _plot_median(
    plot_median_line: bool,
    median_shap_df: pd.DataFrame,
    feature_col: str,
    segment_col: Union[str, None] = None,
    color_dir: Union[dict, None] = None,
) -> None:
    """Plots the median SHAP values on top of scatter plot.
    Plot each group separately if segment group (segment_col) is assigned.

    Args:
        plot_median_line (boolean): Whether to show the median line for shap values.
        median_shap_df (pd.DataFrame): df with columns [feature-name, flag-name, median-SHAP-value]
                                         if segment_col is assigned,
                                         else df with columns [feature-name, median-SHAP-value].
        feature_col (str): Name of the column that we want to visualize
        segment_col (str): Name of the column we use to color-code scatter plot.
                           Doesn't need to be a feature in the model.
        color_dir (dict or None): a dictionary that maps distinct values of the flag feature to colors,
                                  only use when segment_col is not None.

    Returns:
        The median line plot with legend.
        Color-coded by segment group (segment_col) if it's assigned.

    """
    if plot_median_line:
        if segment_col is None:
            plt.plot(
                feature_col,
                "Median",
                data=median_shap_df,
                label="Median Contribution for {}".format(feature_col),
            )
        else:
            # Re-create color dictionary when segment_col is assigned,
            # but no color_dir is passed
            if color_dir is None:
                # Create a list of color based on values of column segment_col
                cmap = matplotlib.cm.get_cmap("Set1")
                color_dir = {}
                for i, cat in enumerate(median_shap_df[segment_col].unique()):
                    color_dir[cat] = cmap.colors[i]

            for cur_cate in median_shap_df[segment_col].unique():
                plt.plot(
                    feature_col,
                    "Median",
                    data=median_shap_df.loc[median_shap_df[segment_col] == cur_cate, :],
                    label="Median Contribution for "
                    + segment_col
                    + "=={}".format(cur_cate),
                    color=color_dir[cur_cate],
                )
        plt.legend()


def _set_axis_limit(
    selected_xlim: Union[List[float], None], selected_ylim: Union[List[float], None]
) -> None:
    """Manually sets the axis limits for x-axis and y-axis.

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


def _plot_histogram(
    plot_histogram: bool, feature_col: str, data_df: pd.DataFrame, nbins: int
) -> None:
    """Plots histogram of the selected feature in the background.

    Args:
        plot_histogram (boolean): If True, a population histogram will be plotted in the background
        feature_col (str): Name of the column that we want to visualize
        data_df (pd.DataFrame): The test data with all columns
        nbins (int): The number of bins in histogram

    Returns:
        Histogram plot in the background.

    """
    if plot_histogram:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.hist(
            data_df[feature_col],
            bins=nbins,
            alpha=0.3,
            label="Histogram Count",
            color="grey",
        )
        plt.legend(loc="upper left")
