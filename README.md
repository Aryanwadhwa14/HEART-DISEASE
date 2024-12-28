# Heart Disease Prediction and Patient Profiling

This project aims to predict the presence of heart disease in patients and create distinct patient profiles based on their risk factors. By combining predictive modeling with clustering techniques, the project provides actionable insights into heart disease risks, helping clinicians and stakeholders better understand and address key risk factors.

---

## Project Overview

- **Objective**: Develop a machine learning model to predict heart disease and cluster patients into meaningful profiles based on their features.
- **Key Features**:
  - Predict the presence of heart disease using classification algorithms and Artificial Neural Networks (ANN).
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
   - Train an Artificial Neural Network (ANN) for improved accuracy and performance.
   - Evaluate models using metrics like Accuracy, Precision, Recall, and ROC-AUC.

3. **Patient Profiling**:
   - Apply clustering techniques (e.g., K-Means) to group patients.
   - Validate clusters using metrics like Silhouette Score and Davies-Bouldin Index.

4. **Risk Scoring**:
   - Combine prediction results and clustering for risk categorization (Low, Medium, High).

5. **Web Dashboard**:
   - Interactive dashboard using Streamlit for data visualization and predictions.

---

## Artificial Neural Network (ANN) Model

- **Architecture**:
  - Input Layer: Takes normalized clinical features such as age, cholesterol, and blood pressure.
  - Hidden Layers: Two fully connected layers with ReLU activation for non-linearity.
  - Output Layer: A single neuron with sigmoid activation to output the probability of heart disease.
  
- **Training Details**:
  - Loss Function: Binary Cross-Entropy.
  - Optimizer: Adam.
  - Epochs: 100 (configurable).
  - Batch Size: 32.

- **Model Code**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define ANN model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
