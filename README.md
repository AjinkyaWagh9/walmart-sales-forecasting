# Walmart Sales Forecasting – Market Trend Analysis

## Project Overview
This project implements an end-to-end machine learning pipeline for market trend analysis using the Walmart Store Sales Forecasting dataset.  
The objective is to predict weekly sales based on historical, categorical, and temporal features.

The focus of this project is not model optimization, but to demonstrate a clear understanding of the complete machine learning workflow, including data exploration, feature engineering, baseline modeling, evaluation, and version control.

---

## Dataset
- **Source**: Kaggle – Walmart Store Sales Forecasting
- **Main file used**: `train.csv`
- **Rows**: 421,570
- **Features**:
  - Store
  - Department
  - Date
  - Weekly_Sales
  - IsHoliday

> Raw data files are not included in the repository and must be downloaded separately.

---

## Project Structure
walmart-sales-forecasting/
├── data/

│   └── raw/               # Not tracked (ignored via .gitignore)
├── src/
│   ├── eda.py             # Exploratory data analysis and visualization
│   ├── train_model.py     # Baseline Linear Regression model
│   └── train_rf.py        # Random Forest model for comparison
├── results/
│   ├── baseline_metrics.txt
│   └── rf_metrics.txt
├── requirements.txt
├── .gitignore
└── README.md

---

## Exploratory Data Analysis (EDA)
Key steps performed:
- Data inspection and validation
- Missing value analysis
- Conversion of date column to datetime format
- Feature extraction from date:
  - Year
  - Month
  - ISO Week number
- Visualization of total weekly sales over time
- Comparison of holiday vs non-holiday sales

### Key Insights
- Weekly sales exhibit strong seasonal patterns.
- Sales peaks occur during holiday periods.
- Holiday weeks have higher average sales, making `IsHoliday` a useful predictive feature.

---

## Feature Engineering
Temporal features extracted from the date column:
- Year
- Month
- Week

These features help capture seasonality and time-based trends in sales data.

---

## Baseline Model
- **Model**: Linear Regression
- **Type**: Regression
- **Target Variable**: Weekly_Sales
- **Features Used**:
  - Store
  - Dept
  - Year
  - Month
  - Week
  - IsHoliday

### Train-Test Split
- 80% training data
- 20% testing data
- Fixed random state for reproducibility

---

## Model Evaluation
Evaluation metrics used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Baseline performance:
- **MAE**: 15,125.90
- **RMSE**: 22,482.37

The baseline model i.e Liner Regression serves as a reference point for future improvements.

### Random Forest Model
- **MAE**: 1,431.91
- **RMSE**: 4,010.74

The Random Forest model demonstrates a significant improvement over the baseline by capturing non-linear patterns and feature interactions present in retail sales data.

---

## Improved Model – Random Forest Regression

To capture non-linear relationships and interactions between features, a Random Forest Regression model was implemented as a comparison to the baseline Linear Regression model.

Random Forest is an ensemble learning method that combines multiple decision trees and is well-suited for complex, real-world datasets such as retail sales data.

### Model Details
- **Model**: Random Forest Regressor
- **Type**: Regression
- **Target Variable**: Weekly_Sales
- **Features Used**:
  - Store
  - Dept
  - Year
  - Month
  - Week
  - IsHoliday

The model was trained using default hyperparameters without tuning, as the goal was model comparison rather than optimization.

### Model Comparison

| Model               | MAE       | RMSE      |
|---------------------|-----------|-----------|
| Linear Regression   | 15,125.90 | 22,482.37 |
| Random Forest       | 1,431.91  | 4,010.74  |

The comparison highlights the importance of model selection for complex forecasting tasks.

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Git & GitHub
- VS Code (macOS)

---

## How to Run
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python src/eda.py
   python src/train_model.py
   python src/train_rf.py
