# Breath-Based Diabetes and Lung Cancer Detection (Deep Learning)

## Problem
Diabetes and lung cancer are often found late. Breath contains volatile organic compounds (VOCs), such as acetone, that change with disease. This project classifies samples as Healthy, Diabetes, or Lung cancer from breath VOC readings using a 1D CNN.

## Data
Simulated VOC sensor data (1,500 samples, 5 features: acetone, isoprene, ethanol, benzene-type VOC, humidity), generated to practice the full pipeline. This is NOT real patient data.

## Steps
1. Data creation and exploration
2. Preprocessing: removed duplicates, filled missing values (median), scaled features, stratified 70/15/15 train/validation/test split
3. Baseline models: Logistic Regression, Random Forest
4. 1D CNN, tuned with filters, dropout, learning rate, and early stopping
5. Evaluation on the test set: accuracy, precision, recall, F1, confusion matrix

## Results
| Model | Test accuracy |
|---|---|
| Logistic Regression |0.8756 |
| Random Forest |0.8933 |
| Tuned 1D CNN |0.8756 |


On this simulated dataset, the tuned CNN performed comparably to the baselines, and Random Forest scored slightly higher. CNN per-class F1: Healthy 0.84, Diabetes 0.91, Lung cancer 0.88.

## Limitations
- Simulated data, so results do not reflect real clinical performance
- Next step: validate on a real breath dataset

## Tools
Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn
