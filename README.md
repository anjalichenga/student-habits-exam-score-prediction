# Student Performance Prediction — Habits vs. Exam Scores

Predicting student exam scores from daily lifestyle habits (study time, sleep,
screen time, mental health, attendance, etc.) using exploratory data analysis
and regression modeling.

## 📌 Overview

This project explores how everyday student habits relate to academic
performance, then builds a model to predict exam scores from those habits.
The goal was to identify which lifestyle factors matter most and quantify
how well they can predict outcomes.

## 📊 Dataset

- **1,000 students**, **16 features** including:
  - Study hours/day, sleep hours, attendance %, exercise frequency
  - Social media hours, Netflix hours, part-time job status
  - Diet quality, mental health rating, parental education level, internet quality
  - Target variable: `exam_score`

## 🔍 Process

1. **Exploratory Data Analysis**
   - Checked for missing values and duplicates, reviewed distributions and
     summary statistics for numeric and categorical features.
   - Visualized relationships between habits and exam scores (histograms,
     boxplots, pairplots, correlation heatmap).

2. **Feature Engineering & Preprocessing**
   - Combined `social_media_hours` + `netflix_hours` into a single
     `entertainment_hours` feature to reduce redundancy.
   - Label-encoded categorical variables (`gender`, `part_time_job`).
   - Standardized features with `StandardScaler` before modeling.

3. **Modeling**
   - Trained and compared four regressors: Linear Regression, Decision Tree,
     Random Forest, and XGBoost.
   - Tuned models with `GridSearchCV` (5-fold CV) over key hyperparameters
     (e.g., Ridge alpha, tree depth, n_estimators, learning rate).
   - Used **RFE (Recursive Feature Elimination)** to identify the top
     predictive features across different model types.
   - Evaluated residual plots to check for bias/heteroscedasticity in predictions.

## Baseline Results

| Model               | MSE    | R² Score |
|---------------------|--------|----------|
| **Linear Regression** | **25.94** | **0.899** |
| XGBoost             | 32.90  | 0.872    |
| Random Forest       | 35.65  | 0.861    |
| Decision Tree       | 80.88  | 0.685    |

Final model after optimising Linear Regression model with GridSearchCV
Best Ridge Alpha: {'alpha': 1.0}
Best R²: 0.90

--- Random Forest Optimized ---
Best Params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
Best R²: 0.86

--- XGBoost Optimized ---
Best Params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
Best R²: 0.88

### Key drivers of exam score (via feature importance / RFE)
- Study hours per day
- Attendance percentage
- Sleep hours
- Mental health rating

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (modeling, GridSearchCV, RFE)
- XGBoost

## 📂 Project Structure

```
├── student_performance.ipynb        # Full analysis & modeling notebook
├── student_habits_performance.csv   # Dataset
└── README.md
```

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook student_performance.ipynb
```

