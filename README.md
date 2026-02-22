# 🎓 Student Performance Predictor

## Overview
Student Performance Predictor is a Streamlit-based web application that predicts student academic performance based on input features such as attendance, study hours, subject scores, and internet access. It utilizes a pre-trained Random Forest machine learning model to classify students into different performance categories (e.g., At-Risk, Average, High-Performing) and assigns them a predicted grade (Grade A to F).

Furthermore, the application analyzes individual student features to generate personalized, actionable study recommendations for improvement.

## Features
- **CSV Data Upload**: Upload a batch of student records via a CSV file.
- **Grade Prediction**: Automatically predict grades (A-F) for each student.
- **Classification**: Categorize students into At-Risk, Average, or High-Performing buckets.
- **Personalized Recommendations**: Get rule-based study recommendations tailored to factors like low attendance, poor subject scores, or lack of internet access.
- **Data Visualization**: View distribution charts of predicted grades and classifications.
- **Export Results**: Download the predictions and classifications as a new CSV file.

## Project Structure
```text
VectoSpace/
├── app.py                   # Main Streamlit application
├── notebooks/               # Jupyter notebooks for data analysis and experiments
│   ├── data_analysis.ipynb
│   └── experiment_*.ipynb
├── src/
│   └── ml/
│       ├── recommender.py   # Rule-based recommendation engine
│       ├── models/          # Saved ML models (random_forest.pkl, feature_names.pkl)
│       └── *.ipynb          # Model training and batch inference notebooks
└── README.md
```

## Setup & Installation
1. Clone the repository.
2. Ensure you have Python installed (preferably matching your model's environment).
3. Install the required dependencies:
   ```bash
   pip install streamlit pandas scikit-learn
   ```
   *(Note: The exact version of scikit-learn should ideally match the one used to train the `random_forest.pkl` model.)*

## Usage
1. Navigate to the project root directory.
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your web browser.
4. Upload a CSV file containing student data with numerical columns corresponding to the model's training features (e.g., `attendance_percentage`, `study_hours`, `math_score`, `science_score`, `english_score`, `internet_access`).
5. View the predictions, distributions, and study recommendations.
6. Download the results as a new CSV for downstream analysis.

## Machine Learning Integration
The core of the prediction lies in a Random Forest classifier located in `src/ml/models/random_forest.pkl`. The model was trained using the features listed in `src/ml/models/feature_names.pkl`. During inference, the application automatically aligns the uploaded CSV dataset's columns to match the expected features, filling any missing required columns with 0 to prevent feature mismatch errors.
