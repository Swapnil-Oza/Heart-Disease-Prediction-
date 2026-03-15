# 🫀 Heart Disease Prediction

A machine learning project that predicts the **severity level of heart disease** in patients (0–4) based on clinical and demographic features. The project includes full data preprocessing, exploratory data analysis, model training with SMOTE balancing, a scikit-learn pipeline, and a Streamlit web application for real-time predictions.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Workflow](#workflow)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

Heart disease is one of the leading causes of death worldwide. This project applies supervised machine learning to classify the **severity of heart disease** across five levels:

| Level | Meaning                  |
|-------|--------------------------|
| 0     | No Heart Disease         |
| 1     | Mild Heart Disease       |
| 2     | Moderate Heart Disease   |
| 3     | Severe Heart Disease     |
| 4     | Critical Heart Disease   |

The final model is deployed as an interactive **Streamlit web app** where users can enter patient details and instantly receive a prediction.

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (multi-center: Cleveland, Hungary, Switzerland, VA Long Beach)
- **File:** `heart_disease_uci.csv`
- **Samples:** 920 patients
- **Features:** 15 input features + 1 target variable (`num`)

### Feature Descriptions

| Feature    | Description                                              |
|------------|----------------------------------------------------------|
| `id`       | Patient ID                                               |
| `age`      | Age of the patient                                       |
| `sex`      | Sex (Male / Female)                                      |
| `dataset`  | Source hospital/dataset                                  |
| `cp`       | Chest pain type                                          |
| `trestbps` | Resting blood pressure (mm Hg)                           |
| `chol`     | Serum cholesterol (mg/dl)                                |
| `fbs`      | Fasting blood sugar > 120 mg/dl (True / False)           |
| `restecg`  | Resting electrocardiographic results                     |
| `thalch`   | Maximum heart rate achieved                              |
| `exang`    | Exercise-induced angina (True / False)                   |
| `oldpeak`  | ST depression induced by exercise relative to rest       |
| `slope`    | Slope of peak exercise ST segment                        |
| `ca`       | Number of major vessels colored by fluoroscopy           |
| `thal`     | Thalassemia type                                         |
| `num`      | **Target** — Heart disease severity (0 = None, 4 = Critical) |

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── heart.ipynb               # Main Jupyter Notebook (EDA, preprocessing, training)
├── app.py                    # Streamlit web application
├── heart_disease_uci.csv     # Dataset (920 patients, 16 features)
├── pipe.pkl                  # Saved ML pipeline (preprocessing + Random Forest)
├── heart.pkl                 # Saved DataFrame (used for dynamic dropdowns in app)
├── cadio.png                 # Banner image for the Streamlit app
└── README.md
```

---

## ✨ Features

- **Exploratory Data Analysis** with Seaborn, Matplotlib, and Plotly (interactive bar charts, correlation heatmap)
- **Smart Missing Value Imputation** using `IterativeImputer` backed by Random Forest for both categorical and continuous features
- **Class Imbalance Handling** via SMOTE (Synthetic Minority Oversampling Technique)
- **Scikit-learn Pipeline** combining MinMaxScaler + SMOTE + Random Forest Classifier
- **Model Evaluation** with accuracy, confusion matrix, classification report, ROC-AUC curve, and learning curves
- **Cross-validation** (5-fold) for robust performance estimation
- **Streamlit Web App** with a sidebar form for real-time, patient-level predictions
- **ZenML Integration** for ML pipeline orchestration

---

## 🔄 Workflow

```
Raw Data (920 rows)
       │
       ▼
Exploratory Data Analysis (EDA)
       │
       ▼
Missing Value Imputation (IterativeImputer + Random Forest)
       │
       ▼
SMOTE Oversampling (balance all 5 severity classes)
       │
       ▼
Train/Test Split (80/20)
       │
       ▼
Pipeline: MinMaxScaler → SMOTE → Random Forest Classifier
       │
       ├── XGBoost (comparison)
       │
       ▼
Evaluation: Accuracy, ROC-AUC, Cross-Validation
       │
       ▼
Export: pipe.pkl + heart.pkl
       │
       ▼
Streamlit App (app.py)
```

---

## 📈 Models & Results

| Model             | Accuracy | Notes                                |
|-------------------|----------|--------------------------------------|
| **Random Forest** | **87%**  | Final model — used in pipeline & app |
| XGBoost           | 86%      | Comparison model                     |

**5-Fold Cross-Validation (Random Forest):**

| Metric                | Value |
|-----------------------|-------|
| Mean CV Score         | 0.81  |
| Std Dev of CV Scores  | 0.05  |

**Random Forest — Per-Class Performance:**

| Class              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| 0 — No Disease     | 0.84      | 0.85   | 0.84     |
| 1 — Mild           | 0.86      | 0.67   | 0.75     |
| 2 — Moderate       | 0.78      | 0.97   | 0.86     |
| 3 — Severe         | 0.88      | 0.86   | 0.87     |
| 4 — Critical       | 0.97      | 0.98   | 0.97     |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Swapnil-Oza/Heart-Disease-Prediction-.git
   cd Heart-Disease-Prediction-
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > If `requirements.txt` is not present, install manually:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit plotly seaborn matplotlib zenml pillow
   ```

---

## 🚀 Usage

### Run the Jupyter Notebook

Explore the full EDA, preprocessing, and model training:

```bash
jupyter notebook heart.ipynb
```

### Run the Streamlit Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Use the **sidebar form** to enter patient details and click **Predict Level** to get the heart disease severity prediction.

---

## 🛠️ Technologies

| Category          | Tools / Libraries                                 |
|-------------------|---------------------------------------------------|
| Language          | Python 3.x                                        |
| Data Handling     | `pandas`, `numpy`                                 |
| Visualization     | `matplotlib`, `seaborn`, `plotly`                 |
| ML & Pipelines    | `scikit-learn`, `xgboost`, `imbalanced-learn`     |
| Imputation        | `IterativeImputer`, `SimpleImputer`, `KNNImputer` |
| Oversampling      | `SMOTE`                                           |
| Model Persistence | `pickle`                                          |
| Web App           | `Streamlit`, `Pillow`                             |
| MLOps             | `ZenML`                                           |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Swapnil Oza**
GitHub: [@Swapnil-Oza](https://github.com/Swapnil-Oza)

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
