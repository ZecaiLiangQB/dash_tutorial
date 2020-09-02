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

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    """Node for training a simple random forest model."""
    context_cols = parameters["context_cols"]
    feature_cols = [x for x in train_x.columns if x not in context_cols]
    model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=100)
    model.fit(train_x[feature_cols], train_y)

    return model


def predict(model: Any, test_x: pd.DataFrame, parameters: Dict[str, Any]) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set."""
    context_cols = parameters["context_cols"]
    feature_cols = [x for x in test_x.columns if x not in context_cols]
    pred_y = model.predict(test_x[feature_cols])

    return pred_y


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Calculate accuracy of predictions
    r2 = r2_score(test_y, predictions)
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", r2 * 100)
