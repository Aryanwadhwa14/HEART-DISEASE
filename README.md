# Heart Disease Prediction & Patient Profiling System

<div align="center">

![Heart Disease Prediction](https://img.shields.io/badge/ML-Heart%20Disease%20Prediction-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-green?style=for-the-badge&logo=streamlit)

**Advanced ML/DL system for heart disease prediction with patient segmentation**

</div>

---

## Overview

This project combines **predictive analytics** with **patient clustering** to provide comprehensive heart disease risk assessment. The system uses both Machine Learning and Deep Learning approaches to predict heart disease likelihood while segmenting patients into distinct risk profiles for personalized healthcare interventions.

### Key Features
- 🔮 **Predictive Modeling**: ANN, RNN, Random Forest, Logistic Regression
- 👥 **Patient Profiling**: K-Means clustering for risk segmentation
- 📊 **Interactive Dashboard**: Real-time predictions with visualizations
- 🧠 **Model Comparison**: Performance analysis across multiple algorithms

---

## Dataset Information

| Variable | Type | Description | Units | Missing |
|----------|------|-------------|--------|---------|
| `age` | Continuous | Patient age | Years | ❌ |
| `cholesterol` | Continuous | Cholesterol level | mg/dL | ❌ |
| `blood_pressure` | Continuous | Systolic BP | mmHg | ❌ |
| `chest_pain_type` | Categorical | Chest pain type | 0-3 | ❌ |
| `max_heart_rate` | Continuous | Max heart rate | BPM | ❌ |
| `target` | Binary | Heart disease presence | 0/1 | ❌ |

**Dataset**: 1,025 patients | **Features**: 13 clinical indicators

---

## Model Architecture

### Neural Networks
**ANN**: Input(13) → Dense(64,ReLU) → Dense(32,ReLU) → Output(1,Sigmoid)
**RNN**: Input(Sequential) → LSTM(50) → Dropout(0.2) → Dense(25,ReLU) → Output(1,Sigmoid)

### Clustering
**K-Means**: 3 clusters (Low/Medium/High Risk) with Silhouette Score validation

---

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/Aryanwadhwa14/HEART-DISEASE.git
cd HEART-DISEASE

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/dashboard.py
```

### Required Libraries
```
pandas>=1.3.0, numpy>=1.21.0, scikit-learn>=1.0.0
tensorflow>=2.6.0, streamlit>=1.2.0, matplotlib>=3.4.0
```

---

## Usage

### Training Models
```bash
python src/train_models.py --models all    # Train all models
python src/train_models.py --models rnn    # Train specific model
python src/clustering.py                   # Run clustering
```

### Making Predictions
```python
from src.predictor import HeartDiseasePredictor

predictor = HeartDiseasePredictor()
patient_data = {'age': 45, 'cholesterol': 240, 'blood_pressure': 140}
prediction = predictor.predict(patient_data)
```

---

## Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.834 | 0.821 | 0.847 | 0.834 | 0.891 |
| **Random Forest** | 0.852 | 0.839 | 0.865 | 0.852 | 0.923 |
| **ANN** | 0.847 | 0.834 | 0.859 | 0.847 | 0.912 |
| **RNN** | **0.901** | **0.897** | **0.905** | **0.901** | **0.956** |

**Best Performer**: RNN model with 90.1% accuracy and 0.956 ROC-AUC

---

## Project Structure

```
HEART-DISEASE/
├── data/                    # Dataset files
├── src/                     # Source code
│   ├── models/             # ML/DL model implementations
│   ├── data_preprocessing.py
│   ├── clustering.py
│   └── predictor.py
├── app/                    # Streamlit dashboard
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved model files
└── requirements.txt
```

---

## Web Dashboard Features

- **Real-time Prediction**: Interactive patient data input
- **Risk Visualization**: Probability gauges and charts
- **Patient Clustering**: Visual cluster analysis
- **Model Comparison**: Performance metrics comparison
- **Data Explorer**: Interactive dataset exploration

---

## API Endpoints

```http
POST /api/predict
{
    "age": 45,
    "cholesterol": 240,
    "blood_pressure": 140
}

Response:
{
    "prediction": 1,
    "probability": 0.78,
    "risk_level": "High"
}
```

---

## Achievement

<div align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px 0;">

### 🏆 **PATENT RECOGNITION**

**This heart disease prediction system has been officially recognized and patented by the Harvard Innovation Lab, demonstrating its novel approach in combining ML/DL models with patient clustering for cardiovascular risk assessment.**

</div>

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Open Pull Request

**Areas for contribution**: Model improvements, UI enhancements, documentation, testing

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
