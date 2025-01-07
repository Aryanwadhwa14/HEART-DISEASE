# Heart Disease Prediction and Patient Profiling

This project aims to predict the presence of heart disease in patients using Machine Learning (ML) and Deep Learning (DL) models while segmenting patients into actionable profiles based on their clinical features. By combining predictive analytics with clustering, the project enhances the understanding of heart disease risks, aiding clinicians and stakeholders in making informed decisions.

---

## **Project Overview**

### Objectives:
- Predict the presence of heart disease using advanced ML and DL algorithms, including Artificial Neural Networks (ANN) and Recurrent Neural Networks (RNN).
- Create distinct patient profiles using clustering methods like K-Means.
- Provide a web-based dashboard for interactive predictions and data exploration.

### Key Features:
1. **Predictive Modeling**:
   - Train classification models (Logistic Regression, Random Forest, ANN, RNN).
   - Evaluate models for optimal performance using metrics like accuracy and ROC-AUC.
2. **Patient Profiling**:
   - Group patients into meaningful clusters using K-Means.
   - Validate clusters with metrics such as Silhouette Score.
3. **Web Dashboard**:
   - Interactive dashboard for user-friendly predictions and visualizations.

---

## **Dataset Details**

- **Dataset Structure**:
  - **IDs**: Unique patient identifiers.
  - **Features**: Clinical measurements like age, cholesterol levels, and blood pressure.
  - **Target**: Binary variable indicating the presence (1) or absence (0) of heart disease.

| **Variable Name**   | **Role**   | **Type**         | **Description**              | **Units**    | **Missing Values** |
|----------------------|------------|------------------|------------------------------|--------------|---------------------|
| `age`               | Feature    | Continuous       | Age of the patient           | Years        | No                  |
| `cholesterol`       | Feature    | Continuous       | Cholesterol level            | mg/dL        | No                  |
| `blood_pressure`    | Feature    | Continuous       | Blood pressure               | mmHg         | No                  |
| `target`            | Target     | Categorical (0/1)| Presence of heart disease    | -            | No                  |

---

## **Project Workflow**

1. **Data Preprocessing**:
   - Handle missing values and normalize the features.
   - Reshape input data for DL models (e.g., ANN and RNN).

2. **Modeling**:
   - **Machine Learning Models**:
     - Logistic Regression
     - Random Forest
   - **Deep Learning Models**:
     - Artificial Neural Network (ANN)
     - Recurrent Neural Network (RNN)

3. **Clustering**:
   - Apply K-Means clustering to group patients based on risk factors.
   - Evaluate cluster quality with Silhouette Score and Davies-Bouldin Index.

4. **Web Dashboard**:
   - Develop an interactive dashboard using Streamlit for predictions and visualizations.

---

## **Deep Learning Models**

### 1. **Artificial Neural Network (ANN)**
- **Architecture**:
  - Input Layer: Clinical features.
  - Hidden Layers: Two fully connected layers with ReLU activation.
  - Output Layer: Single neuron with sigmoid activation for binary classification.

- **Training**:
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
  - Epochs: 50

- **Performance**:
  - Accuracy: ~47%
  - ROC-AUC: 0.50

### 2. **Recurrent Neural Network (RNN)**
- **Architecture**:
  - Input Layer: Features reshaped to mimic sequential data.
  - LSTM Layer: Processes temporal patterns in the input.
  - Dense Layers: Outputs binary predictions.

- **Training**:
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
  - Epochs: 100

- **Performance**:
  - Accuracy: ~90%
  - ROC-AUC: 0.9

---

## **How to Run**

### Prerequisites:
- Python 3.7 or above
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `streamlit`

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/heart-disease-prediction.git
   cd heart-disease-prediction
