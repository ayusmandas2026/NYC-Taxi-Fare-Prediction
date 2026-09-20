# Capstone Project Report: New York City Taxi Fare Prediction Using Deep Feedforward Neural Networks

**Course:** Student Development Project (SDP Project 2)  
**Author:** Ayusman  
**Domain:** Deep Learning, Applied Machine Learning, Urban Mobility Systems  
**Date:** September 2026  

---

## Abstract

Predicting taxi fares in high-density metropolitan areas like New York City is a complex non-linear regression challenge. Fare pricing is influenced by non-euclidean route geometry, diurnal traffic congestion, regulatory surcharges, and macroeconomic inflation. In this project, an end-to-end deep learning pipeline was engineered to predict taxi trip fares using historical New York City Taxi & Limousine Commission (TLC) trip records. The raw dataset, comprising over 55 million transactions (~5.7 GB), was processed using memory-bounded uniform reservoir sampling to extract an unbiased representative sample of 500,000 records. An extensive anomaly detection framework was implemented to identify and rectify inverted coordinates, remove mathematically invalid records, and sanitize fare/passenger counts. Using great-circle spherical trigonometry (Haversine formula) and temporal feature decomposition, an 11-dimensional feature space was generated. A 4-layer Deep Feedforward Neural Network (Multilayer Perceptron) was built, tuned across multiple loss formulations (MSE, MAE, Huber), and optimized through hyperparameter search. On an untouched held-out test dataset of 75,000 records, the final deep neural network achieved a **Mean Absolute Error (MAE) of \$1.8705**, a **Root Mean Squared Error (RMSE) of \$4.6504**, and an **$R^2$ score of 0.7804**, significantly outperforming the Ordinary Least Squares (OLS) Linear Regression baseline (MAE \$2.4256, $R^2$ 0.7219). Finally, an enterprise-grade interactive deployment was built using Streamlit and PyDeck 3D geospatial visualization, supporting single-trip estimation with TLC regulatory corridor alerts and high-throughput batch simulation.

---

## 1. Introduction & Problem Statement

Urban mobility and on-demand transit systems require reliable, transparent pricing engines. Traditional taxicab pricing in New York City is regulated by the Taxi & Limousine Commission (TLC), which employs meter rates based on initial flag-drop, elapsed distance, stop-and-go idle time, peak hour surcharges, and corridor-specific flat rates (such as between Manhattan and John F. Kennedy International Airport).

While formal meter rate formulas exist, actual fare amounts recorded in historical trip logs exhibit severe non-linear variance due to:
1. **Traffic Congestion Delays:** Manhattan traffic varies drastically by hour of day and day of week.
2. **Geographic Heterogeneity:** Surcharges for inter-borough trips, bridge and tunnel tolls, and airport corridors.
3. **Data Quality Anomalies:** Raw telematics data contains sensor errors, zero coordinates, coordinate inversions, and negative meter entries.

The primary objective of this project is to construct a production-ready deep learning pipeline capable of modeling non-linear trip dynamics, accurately estimating taxi fares, and serving real-time predictions through an accessible web interface.

---

## 2. Dataset & Big Data Preprocessing

### 2.1 Dataset Profile
The underlying dataset originates from the Kaggle NYC Taxi Fare Prediction benchmark:
- **Raw Training Set:** 55,423,856 observations (~5.69 GB CSV)
- **Evaluation Test Set:** 9,914 observations (`test.csv`)
- **Raw Attributes:** `key`, `fare_amount` (target), `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `passenger_count`.

### 2.2 Uniform Reservoir Sampling
Loading 5.7 GB of tabular data simultaneously exceeds standard RAM limits. Rather than taking a naive slice of the initial rows (which introduces strong temporal bias), an **unbiased uniform reservoir sampling** algorithm was executed in streaming chunks of 100,000 rows using a seeded Mersenne Twister / PCG64 random generator:

$$\text{Probability of selection for any record } i = \frac{k}{N} = \frac{500,000}{55,423,856} \approx 0.902\%$$

This yielded a representative working sample of **500,000 observations** spanning the complete temporal distribution (2009 through 2015).

### 2.3 Anomaly Detection & Data Sanitization
Exploratory data analysis revealed multiple telematics anomalies that would degrade neural network convergence:
1. **Zero & Null Coordinates:** 186 records contained $(0.0, 0.0)$ latitude and longitude (Null Island). These were purged.
2. **Coordinate Swapping:** Due to telemetry recording errors, several records had inverted coordinates (latitude $\in [-75, -73]$, longitude $\in [40, 42]$). These confirmed interchanges were systematically swapped back to their correct orientation.
3. **Bounding Box Enforcement:** Legitimate NYC metropolitan trips were strictly bounded:
   $$40.0^\circ \le \text{Latitude} \le 42.0^\circ, \quad -75.0^\circ \le \text{Longitude} \le -73.0^\circ$$
4. **Target Sanitization:** Negative fares and zero fares were removed ($\text{fare\_amount} \ge \$2.50$, the legal NYC TLC minimum flag drop).
5. **Passenger Constraints:** Records with passenger counts outside $[1, 6]$ were eliminated.

---

## 3. Feature Engineering & Preparation

### 3.1 Great-Circle Haversine Distance
Because Euclidean distance on spherical coordinates introduces significant latitude-dependent distortion, the great-circle spherical distance was computed using the Haversine formula:

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$c = 2 \arcsin\left(\sqrt{a}\right)$$
$$d = R \cdot c$$

Where $\phi$ represents latitude, $\lambda$ represents longitude in radians, and $R = 6,371.0\text{ km}$ represents Earth's volumetric mean radius.

### 3.2 Temporal Decomposition
Timestamps were decomposed into 5 distinct cyclical and ordinal features:
- `year`: Captures macroeconomic inflation and the official September 2012 TLC base rate hike.
- `month`: Captures seasonal travel patterns and tourism volume.
- `day`: Captures billing cycles and end-of-month mobility.
- `hour`: Captures diurnal rush hour traffic patterns (4 PM – 8 PM) and overnight rates (8 PM – 6 AM).
- `day_of_week`: Differentiates weekday business commuting from weekend nightlife.

### 3.3 Feature Normalization & Leakage Control
The dataset was partitioned into:
- **Training Set (70%):** 345,000 records
- **Validation Set (15%):** 75,000 records
- **Held-Out Test Set (15%):** 75,000 records

To prevent data leakage, a Scikit-Learn `StandardScaler` was fitted **exclusively on the training split**:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
The fitted scaler parameters were serialized to `MODELS/feature_scaler.pkl` and applied statically to validation, test, and inference pipelines.

---

## 4. Model Architecture & Experimental Design

### 4.1 Baseline: Ordinary Least Squares (OLS) Regression
A multiple linear regression baseline was established using all 11 normalized features to quantify the performance gain delivered by deep neural networks.

### 4.2 Deep Feedforward Neural Network (Multilayer Perceptron)
The deep learning architecture was implemented in TensorFlow 2.x / Keras with the following topological design:

```
[Input Layer: 11 Features]
        │
        ▼
[Dense Layer 1: 128 Units, ReLU Activation] ── (1,536 Parameters)
        │
        ▼
[Dense Layer 2: 64 Units,  ReLU Activation] ── (8,256 Parameters)
        │
        ▼
[Dense Layer 3: 32 Units,  ReLU Activation] ── (2,080 Parameters)
        │
        ▼
[Output Layer: 1 Unit,    Linear Activation] ── (33 Parameters)
```

- **Total Parameters:** 35,717 (11,905 Trainable parameters, 23,812 Optimizer state variables)
- **Activation Function:** Rectified Linear Unit ($\text{ReLU}(x) = \max(0, x)$)
- **Optimization Algorithm:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$)
- **Batch Size:** 1,024
- **Epochs:** 30

### 4.3 Comparative Loss Function Study
Three distinct loss functions were systematically trained and evaluated across identical validation splits:
1. **Mean Squared Error (MSE):** $\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum (y - \hat{y})^2$
2. **Mean Absolute Error (MAE):** $\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum |y - \hat{y}|$
3. **Huber Loss:** Quadratic for small errors ($|y - \hat{y}| \le \delta$) and linear for large deviations ($|y - \hat{y}| > \delta$), mitigating extreme telematics outliers.

### 4.4 Hyperparameter Optimization Matrix
A grid of 5 architectural and optimization configurations was evaluated over 30 epochs:
- Baseline: $[128, 64, 32]$, $\alpha = 0.001$, Batch = 1024
- Higher Learning Rate: $[128, 64, 32]$, $\alpha = 0.003$, Batch = 1024
- Lower Learning Rate: $[128, 64, 32]$, $\alpha = 0.0003$, Batch = 1024
- Smaller Batch Size: $[128, 64, 32]$, $\alpha = 0.001$, Batch = 512
- Expanded Depth & Width: $[256, 128, 64]$, $\alpha = 0.001$, Batch = 1024

---

## 5. Results & Empirical Analysis

### 5.1 Model Performance Benchmarks

| Model Specification | Dataset Split | Objective / Loss | MAE (\$) | RMSE (\$) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression Baseline** | Validation | OLS | 2.4256 | 5.1459 | 0.7219 |
| **Initial DNN (MSE)** | Validation | MSE | 1.8537 | 3.9791 | 0.8337 |
| **DNN Variant (MSE Loss)** | Validation | MSE | 1.8227 | 3.8551 | **0.8439** |
| **DNN Variant (MAE Loss)** | Validation | MAE | **1.6228** | 4.2061 | 0.8142 |
| **DNN Variant (Huber Loss)** | Validation | Huber ($\delta=1.0$) | 1.6327 | 4.1713 | 0.8173 |
| **Final Selected Model** | **Held-Out Test Set** | **MSE** | **1.8705** | **4.6504** | **0.7804** |

### 5.2 Hyperparameter Tuning Evaluation

| Configuration | Architecture | Learning Rate | Batch Size | MAE (\$) | RMSE (\$) | $R^2$ Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | $[128, 64, 32]$ | 0.0010 | 1024 | \$1.8472 | \$3.8719 | 0.8426 | 36.5 s |
| **Higher LR** | $[128, 64, 32]$ | 0.0030 | 1024 | \$1.7844 | \$3.9855 | 0.8332 | 30.6 s |
| **Lower LR** | $[128, 64, 32]$ | 0.0003 | 1024 | \$1.9347 | \$4.0400 | 0.8286 | 31.4 s |
| **Smaller Batch** | $[128, 64, 32]$ | 0.0010 | 512 | \$1.8218 | \$3.9957 | 0.8323 | 46.0 s |
| **Larger Network** | $[256, 128, 64]$ | 0.0010 | 1024 | **\$1.7432** | \$4.0256 | 0.8298 | 57.0 s |

### 5.3 Key Empirical Observations
1. **Deep Learning Superiority:** The DNN achieved a **23.5% reduction in MAE** and a **25.1% reduction in RMSE** over the Linear Regression baseline, proving that non-linear geographical coordinate interactions cannot be captured by linear hyperplanes.
2. **Loss Function Tradeoffs:**
   - Training with **MAE Loss** yielded the lowest absolute dollar error (\$1.6228) because it treats outlier spikes linearly.
   - Training with **MSE Loss** achieved the highest $R^2$ fit (0.8439) and lowest variance penalties (RMSE \$3.8551).
3. **Residual Normality:** Test set residual analysis demonstrated a near-zero mean residual ($-0.2532$) and a median residual of $-0.5778$, confirming absence of systematic under-prediction or over-prediction bias.

---

## 6. Model Explainability & Permutation Feature Importance

To evaluate the black-box decision logic of the neural network, **Permutation Feature Importance** was computed across the holdout test set:

| Rank | Feature Name | Mean Absolute Impact | Relative Importance (%) |
| :---: | :--- | :---: | :---: |
| 1 | `distance_km` | +\$4.67 | **24.77%** |
| 2 | `dropoff_longitude` | +\$3.16 | **16.74%** |
| 3 | `dropoff_latitude` | +\$2.89 | **15.34%** |
| 4 | `pickup_longitude` | +\$2.74 | **14.54%** |
| 5 | `pickup_latitude` | +\$2.18 | **11.57%** |
| 6 | `year` | +\$1.28 | **6.79%** |
| 7 | `hour` | +\$0.80 | **4.24%** |
| 8 | `day_of_week` | +\$0.42 | **2.22%** |
| 9 | `month` | +\$0.38 | **2.03%** |
| 10 | `day` | +\$0.17 | **0.91%** |
| 11 | `passenger_count` | +\$0.16 | **0.84%** |

### Insights:
- **Spatial Coordinates Dominate (58.2% Combined):** While Haversine distance is the single most important individual feature (24.8%), the 4 coordinate variables account for over 58% of predictive power. This reflects borough premiums, bridge tolls, and airport flat-fare zones.
- **Inflationary Sensitivity (`year`, 6.8%):** The neural network successfully internalized the September 2012 TLC rate restructuring.
- **Passenger Count Invariance (0.84%):** In New York City, taxicab meters charge for the vehicle rather than per seat; the model appropriately allocated minimal marginal weight to passenger count.

---

## 7. Interactive Production Deployment

The serialized model (`final_dnn.keras`), scaler (`feature_scaler.pkl`), and metadata schema (`feature_config.json`) were integrated into an enterprise Streamlit dashboard (`DEPLOYMENT/app.py`):

1. **Human-Centric Route Planning:** Offers instant selection of 16 landmark destinations (Times Square, JFK Airport, Grand Central, Wall Street, LaGuardia) and a **1-click Route Swap (⇄)** button.
2. **Interactive 3D Geospatial Cartography (PyDeck):** Renders dynamic 3D arc trajectories and origin/destination markers with daytime light and midnight slate themes.
3. **TLC Regulatory Corridor Alerts:** Automatically detects and displays airport corridor rules (such as JFK flat fares).
4. **Itemized Fare Receipts:** Displays the fare, $\pm \$1.87$ confidence interval, direct distance, estimated traffic transit time, and provides a downloadable `.txt` ride invoice.
5. **High-Throughput Batch Processing:** Supports real-time CSV uploads or random sampling from unseen Kaggle test partitions with summary KPI aggregations and downloadable CSV predictions.
6. **Cloud Readiness:** Configured with a root `requirements.txt` and `.gitignore` for zero-configuration deployment to **Streamlit Community Cloud** or **Docker**.

---

## 8. Limitations & Future Work

1. **Routing Distance vs. Haversine Distance:** Haversine computes great-circle straight-line distance. Integrating a street-network graph routing API (OpenStreetMap / OSRM) would provide turn-by-turn road network distances.
2. **Real-Time Weather Integration:** Severe rain and snow significantly alter NYC taxi demand and traffic delay surcharges.
3. **Tree-Based Ensembling:** Future iterations could combine the Deep Neural Network with a LightGBM regressor to form a weighted stacking ensemble.

---

## 9. Conclusion

This project demonstrates a rigorous, end-to-end applied deep learning workflow. From handling 5.7 GB of big data using streaming reservoir sampling to designing spherical trigonometric features, comparing loss functions, and conducting residual diagnostics, the developed Deep Feedforward Neural Network achieved strong predictive accuracy (MAE \$1.87, $R^2$ 0.7804). The final Streamlit deployment bridges the gap between academic machine learning research and real-world urban transit software engineering.
