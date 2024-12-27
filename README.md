# Heart Disease Prediction and Patient Profiling

This project aims to predict the presence of heart disease in patients and create distinct patient profiles based on their risk factors. By combining predictive modeling with clustering techniques, the project provides actionable insights into heart disease risks, helping clinicians and stakeholders better understand and address key risk factors.

---

## Project Overview

- **Objective**: Develop a machine learning model to predict heart disease and cluster patients into meaningful profiles based on their features.
- **Key Features**:
  - Predict the presence of heart disease using classification algorithms.
  - Cluster patients into distinct profiles based on shared characteristics.
  - Analyze feature importance to understand key risk factors.
  - Provide a web-based dashboard for interactive analysis and risk scoring.

---

## Dataset Information

- **Dataset Source**: [Specify source or "Uploaded Dataset"]
- **Structure**:
  - **IDs**: Unique patient identifiers.
  - **Features**: Clinical measurements like age, cholesterol levels, and blood pressure.
  - **Target**: Binary variable indicating the presence (1) or absence (0) of heart disease.

### Metadata
| **Variable Name** | **Role**     | **Type**         | **Description**                 | **Units**     | **Missing Values** |
|--------------------|--------------|------------------|---------------------------------|---------------|---------------------|
| `age`             | Feature      | Continuous       | Age of the patient             | Years         | No                  |
| `cholesterol`     | Feature      | Continuous       | Cholesterol level              | mg/dL         | No                  |
| `blood_pressure`  | Feature      | Continuous       | Blood pressure                 | mmHg          | No                  |
| `target`          | Target       | Categorical (0/1)| Presence of heart disease       | -             | No                  |

---

## Workflow

1. **Data Preprocessing**:
   - Handle missing values and outliers.
   - Normalize or standardize features for consistency.

2. **Heart Disease Prediction**:
   - Train classification models (e.g., Logistic Regression, Random Forest).
   - Evaluate models using metrics like Accuracy, Precision, Recall, and ROC-AUC.

3. **Patient Profiling**:
   - Apply clustering techniques (e.g., K-Means) to group patients.
   - Validate clusters using metrics like Silhouette Score and Davies-Bouldin Index.

4. **Risk Scoring**:
   - Combine prediction results and clustering for risk categorization (Low, Medium, High).

5. **Web Dashboard**:
   - Interactive dashboard using Streamlit for data visualization and predictions.

---

## How to Use

### Prerequisites
- Python 3.7 or above
- Jupyter Notebook (for local execution)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Aryanwadhwa14/HEART-DISEASE
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the notebook for analysis and model training:
   ```bash
   jupyter notebook HEART_DISEASE_DATASET.ipynb
4.Launch the Streamlit dashboard: 
  ```bash
  streamlit run heart_disease_dashboard.py
5.View predictions and profiles through the interactive web interface.
