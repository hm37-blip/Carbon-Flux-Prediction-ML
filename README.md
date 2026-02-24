
# Ecosystem Carbon Exchange (NEE) Prediction via Machine Learning

## Project Background
This project utilizes high-frequency flux data from the **US-Bo1 (Bondville)** site, part of the AmeriFlux network, to predict Net Ecosystem Exchange (NEE) using machine learning algorithms. In the context of "Carbon Neutrality," accurately forecasting the carbon sequestration and emission of agricultural ecosystems is crucial for understanding and mitigating global climate change.

## Data Pre-processing Pipeline
To help AI models better capture complex natural dynamics, we performed meticulous data cleaning and feature engineering:

### 1. Cyclical Encoding for Temporal Features
Natural processes are inherently periodic. Standard numerical representations of time (e.g., 0-23 hours) fail to represent the proximity between 11 PM (23:00) and Midnight (00:00).
* **Solution**: I mapped temporal features (Hour of Day, Day of Year) onto a 2D unit circle using **Sine** and **Cosine** transformations.
* **Objective**: To empower the model with an intuitive understanding of "Diurnal Cycles" and "Seasonal Successions."

### 2. Meteorological Feature Selection & Imputation
The following key environmental drivers were selected as input features:
* **TA (Air Temperature)**, **SW_IN (Incoming Shortwave Radiation)**, **VPD (Vapor Pressure Deficit)**, **P (Precipitation)**, and **Soil Moisture (SWC)**.
* **Technique**: I applied **Forward Fill (ffill)** to repair short-term missing data caused by sensor malfunctions and excluded samples where the target variable (NEE) was missing.
                           **Median Imputation**: For any remaining gaps that `ffill` couldn't solve (e.g., missing data at the very beginning of the series), we used the column-wise **median**.
                           These steps ensure no `NaN` values are fed into the machine learning models.
* **Target Handling**: We strictly excluded samples where the target variable (**NEE**) was missing to avoid training on noise.

### 3. Quality Control
* Removed invalid flag values (-9999).
* Built a high-frequency time-series dataset indexed by UTC timestamps using the `pandas` library.


