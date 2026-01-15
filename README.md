# Student Performance Prediction using Machine Learning

This project predicts a student’s **final exam grade (G3)** using Machine Learning.  
It uses student academic, demographic, and lifestyle data to build a predictive model.

The goal is to help identify students at risk and understand which factors influence performance.

---

## Dataset

The project uses two datasets:

- `student-mat.csv` (Mathematics students)
- `student-por.csv` (Portuguese students)

Both datasets are merged to create a larger training dataset.

Source: UCI Machine Learning Repository — Student Performance Dataset.

---

## Features Used

The model uses multiple factors including:

- Age  
- Gender  
- Study time  
- Past failures  
- Absences  
- Health  
- Internet access  
- First exam grade (G1)  
- Second exam grade (G2)  
- Family, school, and background attributes (one-hot encoded)

Target variable:
- **G3** (Final exam grade)

---

## Machine Learning Model

The model used is:

**Random Forest Regressor**

It was chosen because:
- It handles non-linear relationships well
- It works well with mixed features
- It is robust to noise

---

## Model Performance

After training:

- **R² Score ≈ 0.81**
- **RMSE ≈ 1.71**

This means the model explains about **81% of the variation** in final grades.

---

## Project Structure

student-performance-ml/
│
├── data/
│ ├── student-mat.csv
│ └── student-por.csv
│
├── models/
│ ├── rf_with_grades.pkl
│ └── feature_columns.pkl
│
├── src/
│ ├── load_data.py
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
│
└── README.md

---

## How to Run the Project

### Step 1 — Install dependencies

pip install pandas numpy scikit-learn joblib

---

### Step 2 — Train the model

python src/train.py


This will:
- Load and merge data
- Preprocess features
- Train the Random Forest model
- Save the trained model and feature layout

---

### Step 3 — Make a prediction

python src/predict.py

You will be asked to enter:
- Age  
- Gender  
- Study time  
- Failures  
- Absences  
- Health  
- Internet access  
- G1  
- G2  

The program will output the **predicted final grade (G3)**.

---

## Example Output

Predicted Final Grade (G3): 11.43

---

## Why G1 and G2 Are Important

G1 and G2 are previous exam scores.  
They are the **strongest predictors** of final performance, so they greatly improve accuracy.

---

## Educational Value

This project demonstrates:
- Data cleaning and merging
- Feature encoding
- Machine learning training
- Model evaluation
- Real-time prediction system

---

## Author

**(Muhammad Ali, Shakeel Ahmad)**   

---

## Submission

This project was developed as part of a **Machine Learning course project** for academic submission.
