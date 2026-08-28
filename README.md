# PM2.5 Prediction Using a Hybrid LightGBM and Quantum SVR Model

##  Overview

This project implements a **hybrid machine learning approach for PM2.5 prediction** by combining the classical **LightGBM** algorithm with **Quantum Support Vector Regression (QSVR)**.

The proposed architecture consists of:

1. **LightGBM** as the primary classical regression model.
2. **Quantum Support Vector Regression (QSVR)** using a fidelity-based quantum kernel.
3. **Quantum Bagging** to generate an ensemble of quantum regression models.
4. **Ridge Regression Meta-Learner** to combine the predictions from LightGBM and QSVR.

The final model is evaluated using **R² (coefficient of determination)** and **RMSE (Root Mean Squared Error)** on the test dataset.

---

##  Model Architecture

The overall workflow can be summarized as follows:

```text
                         PM2.5 Dataset
                              │
                              ▼
                     Data Preprocessing
                              │
                              ▼
                       Train/Test Split
                         80% / 20%
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
             LightGBM                  Quantum SVR
                                            │
                                            ▼
                                   Quantum Feature Map
                                            │
                                            ▼
                                  Fidelity Quantum Kernel
                                            │
                                            ▼
                                      SVR Regression
                                            │
                                            ▼
                                     Quantum Bagging
                                      (5 Estimators)
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                       Base Model Outputs
                              │
                              ▼
                     Ridge Meta-Learner
                              │
                              ▼
                    Final PM2.5 Prediction
```

---

##  Dataset

The dataset used in this project is:

```text
Data_Udara_Pekanbaru_Normalisasi.csv
```

The dataset contains normalized air-quality and meteorological variables, with the target variable:

```text
PM2.5
```

The dataset is already normalized to a **0–1 scale**.

Several pollutant variables, including `SO2`, `O3`, `NO2`, and `HC`, are excluded from the modeling process.

### Features

The LightGBM model uses the following nine features:

| No. | Feature        |
| --: | -------------- |
|   1 | `PM10`         |
|   2 | `CO`           |
|   3 | `pm2.5_lag1`   |
|   4 | `pm10_lag1`    |
|   5 | `pm2.5_trend3` |
|   6 | `pm10_trend3`  |
|   7 | `suhu_avg`     |
|   8 | `curah_hujan`  |
|   9 | `angin_max`    |

Target variable:

```text
PM2.5
```

These features and the target are extracted directly from the dataset before the train/test split.

---

## ⚙️ Technologies and Libraries

The implementation is written in **Python** and uses the following main libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* LightGBM
* Qiskit
* Qiskit Machine Learning
* Qiskit Algorithms

The quantum component uses:

* `ZZFeatureMap`
* `StatevectorSampler`
* `ComputeUncompute`
* `FidelityQuantumKernel`

These components are used to construct the quantum feature map and fidelity-based quantum kernel.

---

## 📦 Installation

The original implementation is designed to run in **Google Colab**.

Install the Qiskit dependencies using:

```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms
```

The versions recorded during the original execution were:

```text
qiskit                  2.4.1
qiskit-machine-learning 0.9.0
qiskit-algorithms       0.4.0
rustworkx               0.17.1
stevedore                5.7.0
```

The installation command is included in the original notebook.

For a complete environment, the following packages are also required:

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm
```

A `requirements.txt` file can therefore contain:

```text
numpy
pandas
matplotlib
scikit-learn
lightgbm
qiskit==2.4.1
qiskit-machine-learning==0.9.0
qiskit-algorithms==0.4.0
```

---

#  How to Run

## 1. Open the Notebook

The original implementation uses Google Colab because it relies on:

```python
from google.colab import files
```

## 2. Install Dependencies

Run:

```python
!pip install qiskit qiskit-machine-learning qiskit-algorithms
```

## 3. Upload the Dataset

The notebook uploads the dataset using:

```python
uploaded = files.upload()
```

The uploaded CSV file is then loaded using Pandas:

```python
df = pd.read_csv(nama_file)
```

The expected dataset is:

```text
Data_Udara_Pekanbaru_Normalisasi.csv
```

---

#  Data Preprocessing

The dataset is divided into training and testing subsets using an **80:20 split**.

The split is performed without shuffling:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

Using `shuffle=False` preserves the original temporal order of the observations.

The original experiment produced:

```text
Training data : 583 days
Testing data  : 146 days
```

This approach is particularly relevant for the time-series structure of the air-quality data.

---

#  Modeling Pipeline

## 1. LightGBM

The first base model is a LightGBM Regressor:

```python
best_lgbm = lgb.LGBMRegressor(
    random_state=42,
    verbose=-1
)

best_lgbm.fit(X_train, y_train)
```

The trained LightGBM model generates predictions for both the training and testing datasets.

---

## 2. Quantum Feature Selection

The QSVR component uses a subset of the available features.

Based on the feature ordering, the selected quantum features correspond to:

```text
pm2.5_lag1
PM10
pm2.5_trend3
pm10_lag1
```

Thus, the quantum model operates on four input features.

---

## 3. Quantum Feature Map

The quantum feature map is constructed using `ZZFeatureMap`.

The configuration uses:

```text
Feature dimension : 4
Repetitions       : 1
Entanglement      : linear
```

Conceptually:

```python
ZZFeatureMap(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)
```

The original implementation uses this configuration for the QSVR quantum kernel.

---

## 4. Fidelity Quantum Kernel

The quantum kernel is constructed using Qiskit's statevector-based components:

```python
sampler = StatevectorSampler()

fidelity = ComputeUncompute(
    sampler=sampler
)

quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)
```

The resulting kernel is then used as a **precomputed kernel** for the SVR model.

---

# 🔎 QSVR Hyperparameter Tuning

The QSVR hyperparameters are tuned using `RandomizedSearchCV`.

The search space includes:

| Parameter | Candidate Values     |
| --------- | -------------------- |
| `C`       | `0.1`, `1`, `10`     |
| `epsilon` | `0.01`, `0.1`, `0.2` |

The tuning process uses:

```python
RandomizedSearchCV(
    SVR(kernel="precomputed"),
    param_distributions={
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.2]
    },
    n_iter=5,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42
)
```

The tuning procedure is performed on a subsample to reduce the computational cost of repeatedly evaluating the quantum kernel.

---

#  Quantum Bagging

After obtaining the QSVR configuration, the quantum model is trained using a **bagging approach**.

The implementation uses:

```text
Number of estimators : 5
Sample size          : 150
```

Five quantum models are trained:

```text
Q-1
Q-2
Q-3
Q-4
Q-5
```

Each model uses a resampled subset of the training data.

The predictions from the five quantum models are then averaged:

```python
pred_train_qsvr = np.mean(
    quantum_preds_train,
    axis=0
)

pred_test_qsvr = np.mean(
    quantum_preds_test,
    axis=0
)
```

The resulting prediction represents the final **Quantum Bagging / QSVR prediction**.

---

#  Manual QSVR Prediction Verification

The implementation also includes a manual verification of the QSVR prediction formula.

The prediction is reconstructed using:

```text
Prediction = Σ(Alpha × Kernel) + Bias
```

The code extracts:

* `dual_coef_` as the alpha coefficients,
* `support_` as the support-vector indices,
* kernel values between the test sample and support vectors,
* `intercept_` as the bias term.

For example, the first quantum model produced:

```text
Bias                  = 0.4025
Number of Support Vectors = 30
```

Example alpha values:

```text
Alpha_1 = -0.2617
Alpha_2 =  0.5998
Alpha_3 =  0.0554
```

The manual calculation and the machine-generated prediction both produced:

```text
0.1958
```

This verification demonstrates that the QSVR prediction can be reconstructed from its learned support-vector representation.

---

#  Ridge Meta-Learner

The predictions generated by the two base models are combined using a **Ridge Regression meta-learner**.

The architecture is:

```text
LightGBM Prediction ───┐
                       ├──> Ridge Meta-Learner ──> Final Prediction
QSVR Prediction ───────┘
```

The meta-learner is configured as:

```python
Ridge(
    alpha=1.0,
    positive=True,
    fit_intercept=True
)
```

The input to the meta-learner consists of two prediction columns:

```python
X_meta_train = np.column_stack(
    (pred_train_lgbm, pred_train_qsvr)
)

X_meta_test = np.column_stack(
    (pred_test_lgbm, pred_test_qsvr)
)
```

The meta-learner is trained against the original training targets.

---

# ⚖️ Meta-Learner Weights

The resulting normalized weights from the meta-learner were:

| Base Model |     Weight |
| ---------- | ---------: |
| LightGBM   | **0.7649** |
| Quantum    | **0.2351** |

Meta-learner bias:

```text
-0.0105
```

This indicates that the final prediction relies more heavily on the LightGBM component than the quantum component in this experiment.

---

# 📈 Experimental Results

The evaluation was performed on the normalized **0–1 scale**.

| Model           |         R² |       RMSE |
| --------------- | ---------: | ---------: |
| LightGBM        |     0.7748 |     0.0387 |
| QSVR            |     0.3612 |     0.0651 |
| Hybrid Ensemble | **0.7779** | **0.0384** |

The final evaluation recorded:

```text
LightGBM:
R²   = 0.7748
RMSE = 0.0387

Hybrid Ensemble:
R²   = 0.7779
RMSE = 0.0384
```

Therefore, the hybrid ensemble achieved a slightly higher R² and lower RMSE than the standalone LightGBM model in the reported experiment.

The QSVR-only evaluation was:

```text
QSVR:
R²   = 0.3612
RMSE = 0.0651
```

---

# 📁 Output Files

The notebook generates several CSV files for analysis and documentation.

## 1. Support Vector Data

```text
Data_Support_Vector_Sirkuit_1_Dengan_Tanggal.csv
```

This file contains information related to the support vectors of the first quantum circuit/model, including the learned coefficients and related prediction information.

---

## 2. LightGBM Predictions

```text
Tabel_Prediksi_LightGBM.csv
```

The file contains:

```text
Indeks_Data_Uji
Target_Aktual_PM2.5
Tebakan_LightGBM
```

It is used to compare the actual PM2.5 values against the LightGBM predictions.

---

## 3. Quantum Predictions

```text
Tabel_Prediksi_Kuantum_Final.csv
```

The file contains:

```text
Indeks_Data_Uji
Waktu_Observasi
Target_Aktual_PM2.5
Prediksi_Arsitektur_Kuantum
```

This output provides the final prediction generated by the quantum architecture.

---

## 4. Final Model Comparison

```text
Tabel_4_13_Komparasi_Final.csv
```

This table contains:

```text
Waktu_Observasi
Target_Aktual_PM2.5
Prediksi_Dasar_LightGBM
Prediksi_Dasar_Kuantum
Prediksi_Akhir_Hibrida
```

It is intended to provide a direct comparison between the actual values, individual base-model predictions, and the final hybrid prediction.

---

# 📂 Recommended Repository Structure

```text
.
├── README.md
├── Data_Udara_Pekanbaru_Normalisasi.csv
│
├── notebook/
│   └── prediksi_pm25_hybrid.ipynb
│
├── output/
│   ├── Data_Support_Vector_Sirkuit_1_Dengan_Tanggal.csv
│   ├── Tabel_Prediksi_LightGBM.csv
│   ├── Tabel_Prediksi_Kuantum_Final.csv
│   └── Tabel_4_13_Komparasi_Final.csv
│
└── requirements.txt
```

---

# 🔁 Complete Workflow

```text
                    ┌──────────────────────┐
                    │        Dataset       │
                    │      PM2.5 Data     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Preprocessing      │
                    │   Feature Selection  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  80/20 Temporal Split│
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌────────────────┐          ┌──────────────────┐
       │    LightGBM    │          │    Quantum SVR   │
       │   9 Features   │          │    4 Features    │
       └───────┬────────┘          └─────────┬────────┘
               │                            │
               │                            ▼
               │                   ┌──────────────────┐
               │                   │  ZZFeatureMap    │
               │                   └─────────┬────────┘
               │                             │
               │                             ▼
               │                   ┌──────────────────┐
               │                   │ Fidelity Quantum │
               │                   │      Kernel      │
               │                   └─────────┬────────┘
               │                             │
               │                             ▼
               │                   ┌──────────────────┐
               │                   │   SVR + Bagging  │
               │                   │    5 Estimators  │
               │                   └─────────┬────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Ridge Meta-Learner │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Final PM2.5 Prediction│
                    └──────────────────────┘
```

---

# 📝 Notes

* The dataset must contain the `PM2.5` target column.
* The required feature columns must also be available.
* The temporal column used for prediction output is `Tanggal`.
* The train/test split uses `shuffle=False`.
* The reported experiment uses data normalized to the **0–1 scale**.
* QSVR uses a **precomputed fidelity quantum kernel**.
* Quantum Bagging uses **5 estimators**.
* The quantum component uses a subset of four features.
* Quantum kernel computation can be computationally expensive, particularly as the dataset size increases.
* The reported performance values are specific to the dataset and experimental configuration used in this implementation.

---

#  Conclusion

This project demonstrates a hybrid approach that combines **classical machine learning** and **quantum machine learning** for PM2.5 prediction.

The reported results show:

```text
LightGBM
R²   = 0.7748
RMSE = 0.0387

QSVR
R²   = 0.3612
RMSE = 0.0651

Hybrid Ensemble
R²   = 0.7779
RMSE = 0.0384
```

The hybrid ensemble slightly outperformed the standalone LightGBM model in the reported experiment, achieving a higher R² and lower RMSE.

The meta-learner assigned a larger contribution to LightGBM (**76.49%**) and a smaller contribution to the quantum model (**23.51%**), indicating that LightGBM provided the stronger predictive signal in this particular experiment.

---

## 👤 Author

**[Kornelius Jonathan Manik]**

This project was developed as an implementation of research on **PM2.5 air-quality prediction using classical and quantum machine learning approaches**.
