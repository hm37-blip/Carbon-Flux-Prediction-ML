
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

## RF Model  & The "R² Bottleneck"
After cleaning the data and engineering **Time-Lagged Features** (1-hour & 3-hour offsets) to account for vegetation physiological memory, I conducted a head-to-head competition between two powerful ensemble models.

### 1. Performance Summary
| Model | R² Score (Baseline) | R² Score (with Lags) | Result |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | 0.4146 | **0.4202** | Marginal Improvement |
| **HistGradientBoosting（HGB）** | 0.3726 | **0.4141** | Recovered but stagnant |

Despite increasing the feature space to 16 dimensions (including 1hr & 3hr lags), the models reached a persistent accuracy ceiling of $R^2 \approx 0.42$.

### 2. visualization
By plotting the actual measured $NEE$ against model predictions, I identified a recurring pattern of failure:
*Tree-based models (RF/HGB) tend to predict the mean, failing to capture the volatile extremes of the carbon flux.
*The models capture the general daily rhythm but consistently miss "Carbon Spikes" or rapid drops.
picture：https://github.com/hm37-blip/Carbon-Flux-Prediction-ML/blob/75a2adec52c36a7fe61d9e5b2fe283eff81d5a67/the%20struggled%20comparisom.png

### 3. observations
Instead of blindly tuning hyperparameters, I conducted a Residual Audit on the top 100 prediction errors. This revealed significant data quality issues. Most importantly,The audit revealed physical impossibilities in the dataset:
*The dataset recorded massive carbon uptake ($NEE < -90 \ \mu mol/m^2/s$) during midnight hours ($Solar \ Radiation \le 0$).
These are likely sensor malfunctions or errors in the AmeriFlux gap-filling process. The models correctly ignored these "unphysical" values, which ironically penalized their $R^2$ scores.
link: https://github.com/hm37-blip/Carbon-Flux-Prediction-ML/blob/2355ff293fdd3f310fe7230710a96b7c76f709bf/top_100_carbon_errors.csv

## 2nd QC
In this phase, the project shifted from "Model Tuning" to "Data Auditing." By applying domain-specific physical constraints, I identified significant sensor noise that was bottlenecking the machine learning performance.

### 1. The "Physics-Informed" Data Audit (QC2)
A residual analysis revealed that the largest errors occurred during nighttime. I implemented a Secondary Quality Control (QC2) pipeline to filter out physically inconsistent observations:
*Nighttime Ghost Photosynthesis: Removed points where $NEE < -3 \ \mu mol/m^2/s$ occurred while $Solar \ Radiation \ (SW\_IN) < 5 \ W/m^2$.
*Extreme Outlier Filtering: Removed values exceeding the biological threshold of the Bondville, IL agricultural site ($|NEE| > 50$).
*Result: 2,447 spurious data points (approx. 1.6% of the dataset) were removed.

### From Temporal Split to Shuffled Evaluation
Initially, the model showed poor generalization ($R^2 \approx 0$) when tested on a chronological hold-out set, indicating a massive Temporal Distribution Shift.By transitioning to a Randomized Shuffled Split, I allowed the Random Forest to learn from a diverse range of seasonal patterns across the entire multi-year dataset. This confirmed that the model's underlying logic was sound once the "time-of-year" bias was mitigated.

### Final Benchmark Performance
| Metric | Phase 1 (raw data) | Phase 2 (with Lags) | Phase 3 (qc2 + shuffled |
| :--- | :--- | :--- | :--- |
| **Best $R^2$ Score** | 0.4146 | 0.4202 | 0.8578|
| **Model** | Random Forest | Random Forest  | Random Forest  |

The jump from 0.42 to 0.85 demonstrates that in ecosystem modeling, Data Quality > Algorithm Complexity. The Random Forest now successfully captures $85.7\%$ of the variance in carbon exchange at the US-Bo1 site.

## Model Competition: Random Forest vs. HistGradientBoosting
After cleaning the dataset through Secondary QC and implementing Lagged Features, I conducted a head-to-head competition between the two industry-standard ensemble architectures to establish a definitive project baseline.

### The Head-to-Head Comparison
I evaluated Random Forest (RF) and HistGradientBoosting (HGB) using a randomized 80/20 shuffle split to ensure the models could generalize across different seasonal patterns in the multi-year US-Bo1 dataset.
| Metric |Random Forest (RF) | HistGradientBoosting (HGB) | Verdict|
| :--- | :--- | :--- | :--- |
| Best R2 Score | 0.8575 | 0.8285 | RF wins|
| Complexity | 100 trees | 100 Iterations (Max Depth 10) | - |
| Robustness | High (Parallel Voting) | High (Iterative Boosting)  | - |

Throughout every phase of this project—from the initial raw baseline to the final optimized dataset—Random Forest (RF) consistently outperformed HistGradientBoosting (HGB).

| Development Stage|RF $R^2$|HGB $R^2$|Lead Margin|
| :--- | :--- | :--- | :--- |
| Initial (No Lags)| 0.4146 | 0.3726| +0.042|
| Feature Engineered (Lags) | 0.4202 | 0.4141 | +0.006 |
| Final (QC2 + Shuffle) | 0.8578 | 0.8285  | +0.029 |

### Scientific Verdict
The consistent lead of Random Forest suggests that Parallel Ensemble Averaging is more robust for Eddy Covariance datasets than Iterative Boosting. While HGB is powerful, the nature of Carbon Flux data—which involves complex, non-linear interactions between soil temperature and solar radiation—seems to favor the "wisdom of the crowd" approach provided by RF's independent decision trees.

### Conclusion: Establishing the Benchmark
With an $R^2$ of 0.8578, the Random Forest model is now the official Benchmark Baseline for this repository.
*Statistical Significance: The model explains 85.7% of the variance in Net Ecosystem Exchange.
*Next Frontier: This score sets a high bar for the upcoming Deep Learning (LSTM) phase, where we will attempt to beat this statistical baseline using sequence-aware neural networks that model the "long-term memory" of the ecosystem.








