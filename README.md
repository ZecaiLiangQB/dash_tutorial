# Dash Demo: interactive SHAP plot

## Menu
interactive plots on a local browser:
- SHAP summary plot
- SHAP partial dependent plot

## Instructions

- Install packages from requirements.txt or manually install (pandas, sklearn, shap, dash)
- Download bike sharing data ("train.csv") to path "data/01_raw/bike_sharing.csv" 
  (https://www.kaggle.com/c/bike-sharing-demand)
- Run all pipelines (de + ds + shap) by "kedro run"
- Start dash app by "python src/dash_demo_shap_plot/pipelines/shap_plot/demo.py"
- Open dash page in a browser at "http://127.0.0.1:8050/" (unless host is specified otherwise)
- Have fun