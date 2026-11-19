# Calorie Burn Prediction | OmahTI Learning Center – DSAI

🏆 **Best Project Award — OmahTI Learning Center (OLC), Data Science & Artificial Intelligence Track**

## Project Overview

This project was developed as the final assignment for the **OmahTI Learning Center (OLC) Data Science & Artificial Intelligence (DSAI) Track**.

The project focuses on predicting calories burned during physical exercise using machine learning, based on individual physiological characteristics and exercise-related factors.

Following the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** methodology, the project covers exploratory data analysis (EDA), data preprocessing, model development, hyperparameter optimization, and performance evaluation.

## Dataset

The dataset contains exercise and physiological information collected from 15,000 individuals, with 9,000 labeled observations used for model development.

The features include:

- **Demographic and physical characteristics:** Gender, Age, Height, and Weight.
- **Exercise-related measurements:** Duration, Heart Rate, and Body Temperature.
- **Target variable:** Calories burned during exercise.

## Methodology

The project follows a CRISP-DM-based machine learning workflow:

1. **Business & Data Understanding:** Defined the prediction objective and examined dataset characteristics.
2. **Exploratory Data Analysis:** Conducted univariate, bivariate, and multivariate analyses to explore feature distributions, relationships, and correlations.
3. **Data Preprocessing:** Applied Interquartile Range (IQR)-based outlier capping, one-hot encoding for categorical features, and an 80:20 training-validation split.
4. **Model Development:** Compared Linear Regression, Random Forest Regressor, and XGBoost Regressor.
5. **Hyperparameter Optimization:** Optimized XGBoost using **Optuna with 1,000 trials**.
6. **Model Evaluation:** Evaluated performance using RMSE, MAE, MSE, R², and 5-fold cross-validation.

## Model Performance

**XGBoost Regressor** achieved the strongest performance among the three evaluated algorithms.

### Reported Final Evaluation Scores

| Evaluation Partition | RMSE |
|---|---:|
| 61% of testing data | **1.5038** |
| 39% of testing data | **1.5169** |

### Notebook Results After Hyperparameter Tuning

| Metric | Score |
|---|---:|
| R² Score | 0.9995 |
| Mean Absolute Error (MAE) | 0.9518 |
| Mean Squared Error (MSE) | 2.0415 |
| 5-Fold Cross-Validation R² | 0.9995 |

The results demonstrate XGBoost's strong predictive performance on this dataset, with substantial improvements following hyperparameter optimization.

## Technologies & Tools

- **Python** — Data analysis and machine learning
- **Pandas & NumPy** — Data manipulation and preprocessing
- **Matplotlib & Seaborn** — Exploratory data analysis and visualization
- **Scikit-learn** — Model development and evaluation
- **XGBoost** — Gradient boosting regression
- **Optuna** — Hyperparameter optimization
- **Kaggle Notebook** — Development environment

## Achievement

🏆 **Awarded Best Project — OmahTI Learning Center, DSAI Track**

This project was recognized as the **Best Project** in the OmahTI Learning Center Data Science & Artificial Intelligence track, reflecting the team's implementation of an end-to-end machine learning workflow, from exploratory analysis to predictive modeling and optimization.

---

*Developed by Group 2 as part of the OmahTI Learning Center (OLC) DSAI Final Project.*
