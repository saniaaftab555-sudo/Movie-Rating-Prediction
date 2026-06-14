# 🎬 Movie Rating Prediction with Python

Predicting movie ratings based on features like **genre, director, and cast** using regression techniques. This project covers the full machine learning workflow — from data cleaning to model evaluation — and uncovers what factors actually influence how a movie is rated.

## 📌 Overview

The goal is to analyze historical movie data and build a model that accurately estimates the rating a movie receives from users or critics. Along the way, this project explores data analysis, preprocessing, feature engineering, and machine learning modeling techniques.

## 📊 Dataset

The dataset includes the following columns:
- Name
- Year
- Duration
- Genre
- Rating (target variable)
- Votes
- Director
- Actor 1, Actor 2, Actor 3

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data manipulation
- **Matplotlib** & **Seaborn** – data visualization
- **Scikit-learn** – machine learning models

## 🔍 Workflow

1. **Data Cleaning** – handled missing values, fixed inconsistent numeric formats, and standardized multi-genre entries
2. **Exploratory Data Analysis (EDA)**
   - Distribution of movie ratings
   - Average rating by genre
   - Rating vs. number of votes
   - Correlation heatmap of numeric features
   - Average rating trend over the years
3. **Feature Engineering**
   - Target encoding for high-cardinality features (Director, Actors)
   - Label encoding for Genre
4. **Model Building**
   - Linear Regression
   - Random Forest Regressor
5. **Evaluation**
   - RMSE, MAE, and R² score comparison
   - Actual vs. Predicted rating plot
   - Feature importance analysis

## 📈 Results

| Model              | RMSE  | MAE   | R²   |
|--------------------|-------|-------|------|
| Linear Regression  | 0.56  | 0.46  | 0.38 |
| Random Forest      | 0.43  | 0.35  | 0.64 |

Random Forest outperformed Linear Regression, capturing non-linear relationships between features like director/actor history and movie ratings.

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python movie_rating_prediction.py
```

> Place your dataset (e.g., `IMDb Movies India.csv`) in the project directory, or the script will generate sample synthetic data automatically.

## 📂 Output

The script generates the following visualizations:
- `rating_distribution.png`
- `avg_rating_by_genre.png`
- `rating_vs_votes.png`
- `correlation_heatmap.png`
- `rating_trend_over_years.png`
- `model_comparison.png`
- `actual_vs_predicted.png`
- `feature_importance.png`

## 🔑 Key Takeaways

- Director and cast history significantly influence predicted ratings
- Number of votes correlates with rating, hinting at popularity bias
- Random Forest handles categorical-heavy data better than Linear Regression for this task

author-Sania Aftab(saniaaftab555-sudo)
