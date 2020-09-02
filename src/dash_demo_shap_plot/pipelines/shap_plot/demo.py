# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Useful links:
- more dash components: https://dash.plotly.com/dash-core-components
- dash data table: https://dash.plotly.com/datatable
- more html components: https://dash.plotly.com/dash-html-components
- more attributes for html tag: https://www.w3schools.com/tags/tag_img.asp
"""

import pandas as pd
import shap
from matplotlib import pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go

from plot_utils import (
    _fig_to_uri,
    plot_shap_dependence_plot_with_interaction,
    plot_shap_dependence_plot_by_segment,
)


############################ data ##############################
# load data (hard coded)
train_x = pd.read_csv("data/05_model_input/train_x.csv")
shap_values = pd.read_csv("data/08_reporting/shap_values.csv")


############################ styles ##############################
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

left_margin = 40
tab_default_style = {"fontSize": 20}
tab_selected_style = {"fontSize": 20, "backgroundColor": "#86caf9"}
dropdown_style = {"fontSize": 16, "width": "50%"}
summary_plot_style = {
    "height": "30%",
    "width": "60%",
}
pdp_plot_stype = {
    "height": "50%",
    "width": "50%",
}

# Set Matplotlib backend to a non-interactive
plt.switch_backend("Agg")


############################ layout ##############################
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        ###############
        # 0.Header of the whole website
        html.Div(style={"width": "100%", "height": 10}),  # space
        html.H2(children="Shap Plots Demo", style={"margin-left": left_margin}),
        html.Div(style={"width": "100%", "height": 10}),  # space
        ###############
        dcc.Tabs(
            id="tabs",
            children=[
                ###############
                # 1. First tab
                dcc.Tab(
                    label="Feature Importance Plot",
                    children=[
                        html.Div(style={"width": "100%", "height": 10}),  # space
                        ###############
                        # 1.1 Dropdown menu
                        html.H5(
                            children="Select plot type for feature importance plot:",
                            style={"margin-left": left_margin},
                        ),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="plot_type",
                                    options=[
                                        {"label": "dot plot", "value": "dot"},
                                        {"label": "bar plot", "value": "bar"},
                                    ],
                                    value="bar",
                                    placeholder="Select plot type for feature importance plot",
                                    style=dropdown_style,
                                ),
                            ],
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        # 1.2 Plot
                        html.Div(
                            children=[
                                html.Img(
                                    id="summary_plot",
                                    src="",
                                    style=summary_plot_style,
                                ),
                            ],
                            style={"text-align": "center"},
                        ),
                        ###############
                    ],
                    style=tab_default_style,
                    selected_style=tab_selected_style,
                ),
                ###############
                # 2. Second tab
                dcc.Tab(
                    label="Partial Dependent Plot",
                    children=[
                        html.Div(style={"width": "100%", "height": 10}),  # space
                        ###############
                        # 2.1 Dropdown menu (feature column)
                        html.H5(
                            children="Select feature column to plot:",
                            style={"margin-left": left_margin},
                        ),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="feature_name_tab2",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in shap_values.columns
                                    ],
                                    value=None,
                                    placeholder="Select feature column to plot on the x-axis",
                                    style=dropdown_style,
                                ),
                            ],
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        html.Div(style={"width": "100%", "height": 10}),  # space
                        # 2.2 Dropdown menu (interaction column)
                        html.H5(
                            children="Select interaction feature to plot:",
                            style={"margin-left": left_margin},
                        ),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="interaction_name",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in shap_values.columns
                                    ],
                                    value="auto",
                                    placeholder="Select interaction feature for color-code",
                                    style=dropdown_style,
                                ),
                            ],
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        # 2.3 Choose to plot median line
                        html.H5(
                            children="Plot median line of SHAP values:",
                            style={"margin-left": left_margin},
                        ),
                        dcc.RadioItems(
                            id="median_line_tab2",
                            options=[
                                {"label": "yes", "value": "yes"},
                                {"label": "no", "value": "no"},
                            ],
                            value="no",
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        # 2.4 PDP plot
                        html.Div(
                            children=[
                                html.Img(
                                    id="PDP_plot_tab2",
                                    src="",
                                    style=pdp_plot_stype,
                                ),
                            ],
                            style={"text-align": "center"},
                        ),
                        ###############
                    ],
                    style=tab_default_style,
                    selected_style=tab_selected_style,
                ),
                ###############
                # 3. Third tab
                dcc.Tab(
                    label="Partial Dependent Plot (by segment)",
                    children=[
                        html.Div(style={"width": "100%", "height": 10}),  # space
                        ###############
                        # 3.1 Dropdown menu (feature column)
                        html.H5(
                            children="Select feature column to plot:",
                            style={"margin-left": left_margin},
                        ),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="feature_name_tab3",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in shap_values.columns
                                    ],
                                    value=None,
                                    placeholder="Select feature column to plot on the x-axis",
                                    style=dropdown_style,
                                ),
                            ],
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        html.Div(style={"width": "100%", "height": 10}),  # space
                        # 3.2 Dropdown menu (interaction column)
                        html.H5(
                            children="Select segment column:",
                            style={"margin-left": left_margin},
                        ),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="segment_name",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in train_x.columns
                                        if train_x[i].nunique() <= 5
                                    ],
                                    value=None,
                                    placeholder="Select segment column",
                                    style=dropdown_style,
                                ),
                            ],
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        # 3.3 Choose to plot median line
                        html.H5(
                            children="Plot median line of SHAP values:",
                            style={"margin-left": left_margin},
                        ),
                        dcc.RadioItems(
                            id="median_line_tab3",
                            options=[
                                {"label": "yes", "value": "yes"},
                                {"label": "no", "value": "no"},
                            ],
                            value="no",
                            style={"margin-left": left_margin},
                        ),
                        ###############
                        # 3.4 PDP plot (by segment)
                        html.Div(
                            children=[
                                html.Img(
                                    id="PDP_plot_tab3",
                                    src="",
                                    style=pdp_plot_stype,
                                ),
                            ],
                            style={"text-align": "center"},
                        ),
                        ###############
                    ],
                    style=tab_default_style,
                    selected_style=tab_selected_style,
                ),
                ###############
            ],
        ),
    ],
)


###################### callbacks for Tab 1 ###########################
# choose plot type from dropdown menu,
# generate summary plot
@app.callback(Output("summary_plot", "src"), [Input("plot_type", "value")])
def _generate_summary_plot(plot_type="dot"):
    feature_cols = shap_values.columns
    shap.summary_plot(
        shap_values.to_numpy(), train_x[feature_cols], plot_type=plot_type, show=False
    )
    plt.tight_layout()
    fig = plt.gcf()
    # Convert plt.figure object to base64 encoding
    # Don't need if use plotly library (plotly.express) to plot
    plotly_fig = _fig_to_uri(fig)
    return plotly_fig


###################### callbacks for Tab 2 ###########################
# choose feature name, interaction name, and whether to plot median line
# generate PDP plot
@app.callback(
    Output("PDP_plot_tab2", "src"),
    [
        Input("feature_name_tab2", "value"),
        Input("interaction_name", "value"),
        Input("median_line_tab2", "value"),
    ],
)
def _generate_pdp_plot(feature_name, interaction_name, median_line):
    if feature_name:
        plot_shap_dependence_plot_with_interaction(
            feature_col=feature_name,
            shap_value_df=shap_values,
            data_df=train_x,
            interaction_col=interaction_name,
            plot_median_line=(median_line == "yes"),
        )
        plt.tight_layout()
        fig = plt.gcf()
        # Convert plt.figure object to base64 encoding
        # Don't need if use plotly library (plotly.express) to plot
        plotly_fig = _fig_to_uri(fig)
        return plotly_fig


###################### callbacks for Tab 3 ###########################
# choose feature name, interaction name, and whether to plot median line
# generate PDP plot
@app.callback(
    Output("PDP_plot_tab3", "src"),
    [
        Input("feature_name_tab3", "value"),
        Input("segment_name", "value"),
        Input("median_line_tab3", "value"),
    ],
)
def _generate_pdp_plot_by_segment(feature_name, segment_name, median_line):
    if feature_name:
        plot_shap_dependence_plot_by_segment(
            feature_col=feature_name,
            shap_value_df=shap_values,
            data_df=train_x,
            segment_col=segment_name,
            plot_median_line=(median_line == "yes"),
        )
        plt.tight_layout()
        fig = plt.gcf()
        # Convert plt.figure object to base64 encoding
        # Don't need if use plotly library (plotly.express) to plot
        plotly_fig = _fig_to_uri(fig)
        return plotly_fig


if __name__ == "__main__":
    app.run_server(debug=True)
