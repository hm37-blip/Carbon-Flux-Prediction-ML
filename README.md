
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

### 4. Clean Data 
* address: https://raw.githubusercontent.com/hm37-blip/Carbon-Flux-Prediction-ML/refs/heads/main/final_cleaned_data.csv

## Model Benchmarking & The "R² Bottleneck"
After cleaning the data and engineering **Time-Lagged Features** (1-hour & 3-hour offsets) to account for vegetation physiological memory, I conducted a head-to-head competition between two powerful ensemble models.

### 1. Performance Summary
| Model | R² Score (Baseline) | R² Score (with Lags) | Result |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | 0.4146 | **0.4202** | Marginal Improvement |
Despite increasing the feature space to 16 dimensions (including 1hr & 3hr lags), the models reached a persistent accuracy ceiling of $R^2 \approx 0.42$.

### 2. visualization
By plotting the actual measured $NEE$ against model predictions, I identified a recurring pattern of failure:
*Tree-based models (RF/HGB) tend to predict the mean, failing to capture the volatile extremes of the carbon flux.
*The models capture the general daily rhythm but consistently miss "Carbon Spikes" or rapid drops.




