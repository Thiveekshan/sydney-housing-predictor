# Sydney Housing Price Predictor

A machine learning mini-project predicting residential sale prices across three
Sydney suburbs — **Randwick**, **Parramatta**, and **Campbelltown** — developed as
part of a Machine Learning unit assessment.

## Project Overview

This repository contains a Decision Tree Regressor trained on 101 manually
collected sold-property records, along with a deployed Streamlit web application
for interactive price prediction.

**Live app:** https://sydney-housing-predictor-226247463.streamlit.app/

## Repository Contents

| File | Description |
|---|---|
| `app.py` | Streamlit web application - loads the trained model and predicts sale price from user input |
| `requirements.txt` | Python dependencies required to run the app |
| `.streamlit/config.toml` | Streamlit theme configuration (colours, fonts) |
| `sydney_housing_model.pkl` | Trained Decision Tree Regressor model (saved with joblib) |

The full project — including the collected dataset and the complete analysis
notebook (data cleaning, EDA, model development, evaluation) — is available via
the OneDrive link included in the accompanying report.

## Model Summary

- **Algorithm:** Decision Tree Regressor (max_depth=3, tuned via 5-fold cross-validation)
- **Training data:** 101 properties sold in Randwick, Parramatta, and Campbelltown
- **Features used:** bedrooms, bathrooms, car spaces, property size (sqm), distance
  to Sydney CBD, sale month, suburb, and property type
- **Performance:** 5-fold CV RMSE ≈ $570,781 (see full report for complete
  evaluation, including comparison against Ridge/Lasso Regression and KNN Regressor)

**Note:** the model can only produce valid predictions for the three suburbs it
was trained on. It does not generalise to other Sydney suburbs.

## Running the App Locally

1. Clone or download this repository, keeping the folder structure intact
   (including the `.streamlit` folder).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. A browser tab will open automatically at `http://localhost:8501`.

## Usage

1. Select a suburb and property type from the dropdown menus.
2. Enter the property's bedrooms, bathrooms, car spaces, size, and distance to
   Sydney CBD.
3. Click **"Predict Sale Price"** to view the estimated value.

## Limitations

This model was trained on a small sample (101 properties) collected over an
approximately 8-month window in 2026. Predictions should be treated as indicative,
particularly for properties that are atypical for their suburb (e.g. unusually
large, small, or premium properties). See the accompanying report (Part 4) for a
detailed discussion of model limitations and prediction failure analysis.

## Author

Developed by Thiveekshan as part of a Machine Learning Mini Project (Distinction
task). Report and full analysis notebook available via the link in the project
submission.
