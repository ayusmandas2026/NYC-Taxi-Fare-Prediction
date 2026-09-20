# 🚕 NYC Taxi Fare Prediction Using Deep Feedforward Neural Networks

An end-to-end deep learning project for predicting New York City taxi trip fares using geospatial and temporal feature engineering, multi-layer Deep Feedforward Neural Networks (DNN), and an interactive Streamlit deployment application.

---

## 📌 Project Overview

Accurate taxi fare estimation is critical for ride-hailing services, urban mobility planning, and passenger budgeting. This project develops a deep learning regression pipeline to predict taxi fares given:
- **Pickup & Drop-off coordinates** (latitude and longitude)
- **Trip timestamp** (pickup date and time)
- **Passenger count**

The model was developed on the benchmark **New York City Taxi Fare Prediction** dataset, incorporating comprehensive exploratory data analysis, geographic anomaly detection and correction, distance feature engineering, loss function comparisons, and hyperparameter tuning.

---

## 🏗️ Architecture & Pipeline

```
[Raw Dataset (5.7GB)] 
        │
        ▼ (Reservoir Sampling: 500,000 rows)
[Data Cleaning & Anomaly Handling]
   ├── Coordinate Range & Swap Corrections
   ├── Zero/Missing Location Removal
   └── Fare & Passenger Count Sanitization
        │
        ▼
[Feature Engineering]
   ├── Haversine Distance (distance_km)
   └── Temporal Extraction (year, month, day, hour, day_of_week)
        │
        ▼ (70% Train / 15% Validation / 15% Test Split)
[Feature Scaling (StandardScaler)]
        │
        ▼
[Deep Feedforward Neural Network]
   ├── Dense(128, activation='relu')
   ├── Dense(64,  activation='relu')
   ├── Dense(32,  activation='relu')
   └── Dense(1,   activation='linear')
        │
        ▼
[Model Evaluation & Inference Application]
   ├── Kaggle Test Predictions & Submission
   └── Streamlit Web Application (DEPLOYMENT/app.py)
```

---

## 📊 Dataset & Preprocessing

- **Dataset Size:** 55+ million raw records (~5.7 GB `train.csv`) and 9,914 test records (`test.csv`).
- **Uniform Reservoir Sampling:** An unbiased random sample of 500,000 records was generated in chunks to fit memory constraints while maintaining the underlying population distribution.
- **Anomaly Corrections:**
  - Filtered coordinates outside the NYC metropolitan bounding box ($40^\circ \le \text{lat} \le 42^\circ$, $-75^\circ \le \text{lon} \le -73^\circ$).
  - Corrected inverted/swapped latitude and longitude records.
  - Eliminated negative/zero fares and invalid passenger counts ($0$ or $>6$).
- **Features Extracted (11 features):**
  - Geospatial: `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `distance_km` (Haversine formula).
  - Temporal: `year`, `month`, `day`, `hour`, `day_of_week`.
  - Operational: `passenger_count`.

---

## 🧠 Modeling & Performance Comparison

### 1. Baseline vs. Deep Neural Network
The deep feedforward neural network significantly outperformed the baseline Ordinary Least Squares (OLS) Linear Regression model:

| Model | Split | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | Validation | \$2.4256 | \$5.1459 | 0.7219 |
| **Initial DNN (MSE Loss)** | Validation | \$1.8537 | \$3.9791 | 0.8337 |
| **DNN (MAE Loss)** | Validation | \$1.6228 | \$4.2061 | 0.8142 |
| **DNN (Huber Loss)** | Validation | \$1.6327 | \$4.1713 | 0.8173 |
| **Final Selected DNN** | **Held-Out Test Set** | **\$1.8705** | **\$4.6504** | **0.7804** |

### 2. Hyperparameter Tuning Experiments (30 Epochs)
| Configuration | Architecture | Learning Rate | Batch Size | MAE ($) | RMSE ($) | $R^2$ | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | [128, 64, 32] | 0.0010 | 1024 | \$1.8472 | \$3.8719 | 0.8426 | 36.5 s |
| **Higher LR** | [128, 64, 32] | 0.0030 | 1024 | \$1.7844 | \$3.9855 | 0.8332 | 30.6 s |
| **Lower LR** | [128, 64, 32] | 0.0003 | 1024 | \$1.9347 | \$4.0400 | 0.8286 | 31.4 s |
| **Smaller Batch** | [128, 64, 32] | 0.0010 | 512 | \$1.8218 | \$3.9957 | 0.8323 | 46.0 s |
| **Larger Network** | [256, 128, 64] | 0.0010 | 1024 | \$1.7432 | \$4.0256 | 0.8298 | 57.0 s |

---

## 📁 Repository Structure

```
├── .gitignore                          # Git exclusions for large datasets and virtualenvs
├── README.md                           # Comprehensive project documentation
├── DATA/
│   └── raw/
│       ├── train.csv                   # Raw training dataset (~5.7 GB, gitignored)
│       ├── test.csv                    # Evaluation test set (9,914 rows)
│       └── sample_submission.csv       # Submission template
├── NOTEBOOK/
│   └── NYC_Taxi_Fare_Prediction.ipynb  # End-to-end Jupyter Notebook (EDA, Modeling, Evaluation)
├── MODELS/
│   ├── final_dnn.keras                 # Trained Keras DNN model artifact
│   ├── feature_scaler.pkl              # Fitted Scikit-Learn StandardScaler
│   └── feature_config.json             # Feature schema and order configuration
├── DEPLOYMENT/
│   ├── app.py                          # Interactive Streamlit web application
│   └── requirements.txt                # Deployment dependencies
└── RESULTS/
    ├── submission.csv                  # Test predictions in Kaggle submission format
    ├── test_predictions.csv            # Detailed test set predictions with features
    ├── model_comparison.csv            # Baseline vs. DNN metrics table
    ├── hyperparameter_tuning_results.csv # Hyperparameter experiment logs
    ├── evaluation_metrics.json         # Structured evaluation summary JSON
    ├── validation_models_comparison.png# Visual comparison of model metrics
    ├── hyperparameter_tuning_comparison.png # Hyperparameter comparison chart
    └── predicted_fares_distribution.png# Distribution plot of test predictions
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r DEPLOYMENT/requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the interactive user interface:

```bash
streamlit run DEPLOYMENT/app.py
```

Once launched, navigate to `http://localhost:8501` in your web browser. You can input custom pickup/dropoff coordinates, passenger count, and datetime to receive instant fare estimates.

---

## 📈 Key Findings & Insights

1. **Nonlinear Relationships:** Taxi fares exhibit pronounced non-linear relationships with geospatial distance and traffic-dependent temporal factors. The DNN achieved a **23.5% reduction in MAE** compared to linear regression.
2. **Geographic Feature Dominance:** The Haversine distance feature is by far the strongest predictor of trip fare, supplemented by pickup/drop-off coordinates that capture borough-specific pricing nuances (e.g., airport flat rates).
3. **Loss Function Tradeoffs:**
   - Training with **MAE Loss** or **Huber Loss** yields lower MAE (\$1.62 vs \$1.82) by mitigating sensitivity to extreme outliers.
   - Training with **MSE Loss** yields lower RMSE (\$3.85 vs \$4.20) and a higher $R^2$ score ($0.8439$).
