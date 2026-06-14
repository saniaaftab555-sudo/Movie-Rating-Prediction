"""
TASK 2 - MOVIE RATING PREDICTION WITH PYTHON
==============================================
Build a model that predicts the rating of a movie based on features
like genre, director, and actors using regression techniques.

Dataset: IMDb Movies India (Name, Year, Duration, Genre, Rating, Votes,
Director, Actor 1, Actor 2, Actor 3)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
DATA_PATH = "IMDb Movies India.csv"   # change to your actual file path

try:
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Generating a synthetic sample dataset instead...")
    rng = np.random.default_rng(42)
    n = 1000
    genres = ["Action", "Drama", "Comedy", "Romance", "Thriller", "Horror", "Sci-Fi"]
    directors = [f"Director_{i}" for i in range(1, 31)]
    actors = [f"Actor_{i}" for i in range(1, 61)]

    df = pd.DataFrame({
        "Name": [f"Movie_{i}" for i in range(n)],
        "Year": rng.integers(1990, 2024, n),
        "Duration": rng.integers(80, 180, n),
        "Genre": rng.choice(genres, n),
        "Votes": rng.integers(50, 50000, n),
        "Director": rng.choice(directors, n),
        "Actor 1": rng.choice(actors, n),
        "Actor 2": rng.choice(actors, n),
        "Actor 3": rng.choice(actors, n),
    })

    # Create a synthetic "true" rating influenced by genre/director/votes
    genre_effect = df["Genre"].map({g: rng.uniform(-1, 1) for g in genres})
    director_effect = df["Director"].map({d: rng.uniform(-0.5, 0.5) for d in directors})
    df["Rating"] = np.clip(
        6.0 + genre_effect + director_effect
        + 0.00002 * df["Votes"]
        + rng.normal(0, 0.4, n),
        1, 10
    ).round(1)

print(df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------------
# Drop rows without a target rating
df = df.dropna(subset=["Rating"])

# Clean numeric columns that may contain text/commas (common in IMDb India CSV)
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )

for col in ["Year", "Duration", "Votes", "Rating"]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

# Fill missing numeric values with median
for col in ["Year", "Duration", "Votes"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
cat_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Use only the first genre if multiple genres are comma-separated
if "Genre" in df.columns:
    df["Genre"] = df["Genre"].astype(str).apply(lambda x: x.split(",")[0].strip())

df = df.dropna(subset=["Rating"]).reset_index(drop=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (GRAPHS)
# ---------------------------------------------------------------

# 3.1 Distribution of Ratings
plt.figure()
sns.histplot(df["Rating"], bins=20, kde=True, color="purple")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=120)
plt.close()

# 3.2 Average Rating by Genre (Top 10)
if "Genre" in df.columns:
    top_genres = df["Genre"].value_counts().head(10).index
    genre_avg = (
        df[df["Genre"].isin(top_genres)]
        .groupby("Genre")["Rating"].mean()
        .sort_values(ascending=False)
    )
    plt.figure()
    sns.barplot(x=genre_avg.values, y=genre_avg.index, palette="viridis")
    plt.title("Average Rating by Genre (Top 10)")
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig("avg_rating_by_genre.png", dpi=120)
    plt.close()

# 3.3 Rating vs Votes (scatter)
if "Votes" in df.columns:
    plt.figure()
    sns.scatterplot(x="Votes", y="Rating", data=df, alpha=0.4, color="teal")
    plt.xscale("log")
    plt.title("Rating vs Number of Votes (log scale)")
    plt.xlabel("Votes (log scale)")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig("rating_vs_votes.png", dpi=120)
    plt.close()

# 3.4 Correlation Heatmap (numeric features)
numeric_cols = [c for c in ["Year", "Duration", "Votes", "Rating"] if c in df.columns]
plt.figure()
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()

# 3.5 Average rating trend over the years
if "Year" in df.columns:
    yearly_avg = df.groupby("Year")["Rating"].mean()
    plt.figure()
    yearly_avg.plot(kind="line", marker="o", color="darkorange")
    plt.title("Average Movie Rating Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig("rating_trend_over_years.png", dpi=120)
    plt.close()

print("\nEDA graphs saved: rating_distribution.png, avg_rating_by_genre.png,"
      " rating_vs_votes.png, correlation_heatmap.png, rating_trend_over_years.png")

# ---------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------
# Encode high-cardinality categorical features using mean-target encoding
# (average rating per category), which works well for Director/Actor columns.
target_enc_cols = [c for c in ["Director", "Actor 1", "Actor 2", "Actor 3"] if c in df.columns]

for col in target_enc_cols:
    mean_rating = df.groupby(col)["Rating"].mean()
    df[col + "_enc"] = df[col].map(mean_rating)

# Label-encode Genre (low cardinality)
if "Genre" in df.columns:
    le_genre = LabelEncoder()
    df["Genre_enc"] = le_genre.fit_transform(df["Genre"])

# ---------------------------------------------------------------
# 5. SELECT FEATURES & TARGET
# ---------------------------------------------------------------
feature_cols = []
for c in ["Year", "Duration", "Votes", "Genre_enc"] + [c + "_enc" for c in target_enc_cols]:
    if c in df.columns:
        feature_cols.append(c)

X = df[feature_cols]
y = df["Rating"]

print("\nFeatures used for modeling:", feature_cols)

# ---------------------------------------------------------------
# 6. TRAIN-TEST SPLIT & SCALING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 7. MODEL TRAINING
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    predictions[name] = y_pred

    print(f"\n{name}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  R2   : {r2:.3f}")

# ---------------------------------------------------------------
# 8. RESULTS COMPARISON GRAPH
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

plt.figure()
results_df[["RMSE", "MAE"]].plot(kind="bar", colormap="Set2")
plt.title("Model Performance Comparison (Lower is Better)")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 9. ACTUAL VS PREDICTED PLOT (Best Model = Random Forest)
# ---------------------------------------------------------------
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_pred = predictions[best_model_name]

plt.figure()
plt.scatter(y_test, best_pred, alpha=0.5, color="indigo")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title(f"Actual vs Predicted Rating ({best_model_name})")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 10. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

plt.figure()
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()

print(f"\nBest performing model: {best_model_name}")
print("\nAll graphs saved successfully in the current directory.")
