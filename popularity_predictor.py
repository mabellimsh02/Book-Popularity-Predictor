"""
INTRO (plain English)
----------------------
The goal: guess whether a book will be "popular" just from its metadata -
things like its genre, its average rating, how long its description is,
and how many other books its author has written. We never look at sales
figures or marketing data, just what you'd know from a book's listing
page.

Since the dataset has no "popular" label, we make one up: a book counts as
popular if it's in the top 25% by number of ratings (ratings_count), which
is a stand-in for "lots of people read and rated this," regardless of
whether they liked it. Because we build that label out of ratings_count,
we throw that column away before training - otherwise the model could
just learn the threshold instead of learning anything real about books.

What the script does, step by step:
  1. Load the data and print basic stats (shape, types, missing values).
  2. Make a couple of exploration charts (EDA).
  3. Engineer features: turn raw columns into useful numbers/categories.
  4. Build the "popular" yes/no label.
  5. Train a simple baseline model (Logistic Regression).
  6. Train a fancier model (Random Forest) and tune it with cross-validation.
  7. Compare both models and plot a confusion matrix + feature importances.

Usage:
    python get_data.py            # one-time download
    python popularity_predictor.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from get_data import DATA_PATH, download

# --- Settings used throughout the script ---------------------------------

RANDOM_STATE = 42       # fixed seed so results are reproducible between runs
TOP_N_CATEGORIES = 15   # how many genres to keep individually (rest -> "Other")
CURRENT_YEAR = 2026     # used to compute how old a book is

# The columns we'll feed into the model, grouped by type because each type
# needs different preprocessing (numbers get scaled, categories get
# one-hot encoded, text gets TF-IDF'd).
NUMERIC_FEATURES = [
    "average_rating",
    "num_pages",
    "book_age",
    "description_length",
    "title_length",
    "author_book_count",
]
CATEGORICAL_FEATURES = ["category_grouped"]
TEXT_FEATURE = "text_blob"


# --- 1. Load & inspect -------------------------------------------------
# Read the CSV and print a quick health check: how many rows/columns, what
# type each column is, and which columns have missing values. This is the
# "look before you leap" step every dataset needs before any modeling.

def load_data() -> pd.DataFrame:
    download()  # make sure data/books.csv exists (downloads it if not)
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    return df


# --- 2. EDA -------------------------------------------------------------
# EDA = Exploratory Data Analysis: just looking at the data visually before
# building anything, to sanity-check assumptions. Here we check two things:
# (a) are ratings mostly high (as Goodreads-style ratings tend to be)?
# (b) how many books are in each genre, and which genres dominate?

def eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: how average ratings are spread out (expect most between 3.5-4.5).
    sns.histplot(df["average_rating"], bins=30, ax=axes[0], color="steelblue")
    axes[0].set_title("Average rating distribution")
    axes[0].set_xlabel("average_rating")

    # Chart 2: how ratings_count is spread out. Popularity is very skewed
    # (a few mega-hits, lots of niche books), so we use a log scale
    # (log1p = log(1 + x), which safely handles books with 0 ratings)
    # to make the shape readable instead of one giant spike near zero.
    sns.histplot(np.log1p(df["ratings_count"]), bins=30, ax=axes[1], color="darkorange")
    axes[1].set_title("Ratings count distribution (log1p scale)")
    axes[1].set_xlabel("log1p(ratings_count)")

    fig.tight_layout()
    fig.savefig("rating_distribution.png", dpi=150)
    plt.close(fig)

    # Chart 3: which genres show up most often, so we know what "Fiction",
    # "Juvenile Fiction", etc. dominance looks like before we bucket genres.
    top_categories = df["categories"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax, color="seagreen")
    ax.set_title("Top 15 genres by book count")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig("category_breakdown.png", dpi=150)
    plt.close(fig)

    print("\nSaved rating_distribution.png and category_breakdown.png")


# --- 3. Feature engineering ---------------------------------------------
# "Feature engineering" just means turning raw columns into inputs that are
# more useful for a model. Raw text and raw genre strings aren't directly
# usable, so here we derive new columns from them.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # A handful of rows are missing genre or author entirely - too little
    # info to use, so we drop them rather than guess.
    df = df.dropna(subset=["categories", "authors"]).copy()

    # Combine title + description into one text blob, since a book's
    # popularity might be hinted at by wording in either place.
    df["text_blob"] = df["title"].fillna("") + ". " + df["description"].fillna("")

    # Word counts as simple numeric signals: a longer, more detailed
    # description or a punchier/longer title might correlate with reach.
    df["description_length"] = df["description"].fillna("").str.split().str.len()
    df["title_length"] = df["title"].fillna("").str.split().str.len()

    # How many other books by this same author are in the dataset - a
    # rough stand-in for "does this author already have a following?"
    df["author_book_count"] = df.groupby("authors")["authors"].transform("count")

    # How old the book is, since older books have had more time to
    # accumulate ratings than something published last year.
    df["book_age"] = CURRENT_YEAR - df["published_year"]

    # There are ~479 unique genre strings, way too many to one-hot encode
    # directly (most would appear only once or twice). So we keep the 15
    # most common genres as-is and lump everything else into "Other".
    top_categories = df["categories"].value_counts().head(TOP_N_CATEGORIES).index
    df["category_grouped"] = df["categories"].where(df["categories"].isin(top_categories), "Other")

    return df


# --- 4. Target ------------------------------------------------------------
# This defines what we're actually trying to predict: a yes/no "popular"
# label, built from the 75th percentile of ratings_count (i.e., top 25%
# most-rated books = popular). See the module docstring for why we use
# ratings_count (reach) instead of average_rating (reception) for this.

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["ratings_count"].quantile(0.75)
    df["popular"] = (df["ratings_count"] >= threshold).astype(int)
    print(f"\nPopularity threshold (75th pct of ratings_count): {threshold:.0f}")
    print(df["popular"].value_counts(normalize=True))
    return df


# --- 5. Model pipeline ----------------------------------------------------
# A ColumnTransformer lets us apply a different preprocessing step to each
# group of columns, all bundled into one object:
#   - numeric columns get scaled (mean 0, std 1) so no single number
#     dominates just because it's on a bigger scale (e.g. num_pages vs.
#     average_rating)
#   - the genre column gets one-hot encoded (turned into 0/1 columns per
#     genre) since it's a category, not a number
#   - the text blob gets TF-IDF'd: turned into up to 300 columns, one per
#     common word/word-pair, weighted by how distinctive that word is

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2)), TEXT_FEATURE),
        ]
    )


def evaluate(name: str, model: Pipeline, X_test, y_test) -> None:
    """Print how well a trained model does on data it hasn't seen (X_test)."""
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    # classification_report gives precision/recall/F1 for both classes -
    # more informative than plain accuracy when classes are imbalanced
    # (here, only ~25% of books are "popular").
    print(classification_report(y_test, preds, target_names=["not popular", "popular"]))
    print(f"F1 (popular class): {f1_score(y_test, preds):.3f}")


def plot_confusion_matrix(model: Pipeline, X_test, y_test, filename: str) -> None:
    """Save a chart showing correct vs. incorrect predictions, by class."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["not popular", "popular"], cmap="Blues", ax=ax
    )
    ax.set_title(filename.replace(".png", "").replace("_", " "))
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, filename: str, top_n: int = 20) -> None:
    """Save a chart of which features the Random Forest relied on most."""
    # Pull the human-readable names for every column the ColumnTransformer
    # produced (numeric names, one per genre, one per TF-IDF word), so the
    # importance chart is labeled with real feature names, not indices.
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["clf"].feature_importances_

    # Sort features by importance (highest first) and keep just the top N.
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(x=importances[order], y=feature_names[order], ax=ax, color="mediumpurple")
    ax.set_title(f"Top {top_n} feature importances (Random Forest)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"\nSaved {filename}")


def main():
    # Steps 1-4: get data ready, with the label attached.
    df = load_data()
    eda(df)
    df = engineer_features(df)
    df = add_target(df)

    # X = the inputs the model sees, y = the answer key it's trying to predict.
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]
    X = df[features]
    y = df["popular"]

    # Split into a training set (80%) the model learns from, and a test
    # set (20%) it never sees until evaluation - this is what tells us if
    # the model actually generalizes instead of just memorizing.
    # stratify=y keeps the 75/25 popular split consistent in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --- Baseline model: Logistic Regression ---
    # Simple, fast, easy to reason about - a good sanity-check model before
    # trying anything fancier. class_weight="balanced" tells it to pay
    # extra attention to the minority ("popular") class, since only ~25%
    # of books are labeled popular.
    baseline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    baseline.fit(X_train, y_train)
    evaluate("Baseline: Logistic Regression", baseline, X_test, y_test)

    # --- Tuned model: Random Forest ---
    # Random Forests can capture more complex, non-linear patterns than
    # logistic regression. Instead of guessing good hyperparameters, we
    # let RandomizedSearchCV try a bunch of combinations and keep whichever
    # scores best on cross-validated F1 (so we're not just tuning to one
    # lucky train/test split).
    rf_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    param_distributions = {
        "clf__n_estimators": [200, 400, 600],      # how many trees in the forest
        "clf__max_depth": [None, 10, 20, 30],      # how deep each tree can grow
        "clf__min_samples_leaf": [1, 2, 4],        # min data points per leaf (higher = less overfitting)
        "clf__max_features": ["sqrt", "log2"],     # how many features each split can consider
    }
    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions,
        n_iter=15,          # try 15 random combinations instead of every combination
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,           # use all CPU cores to speed this up
    )
    search.fit(X_train, y_train)
    print("\nBest RF params:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.3f}")

    # Grab the best-performing model the search found, and check it against
    # the held-out test set (data it has genuinely never seen).
    best_rf = search.best_estimator_
    evaluate("Tuned: Random Forest", best_rf, X_test, y_test)

    # Save charts so we can see *how* it's right/wrong, and *what* it's
    # actually paying attention to.
    plot_confusion_matrix(best_rf, X_test, y_test, "confusion_matrix.png")
    plot_feature_importance(best_rf, "feature_importance.png")


if __name__ == "__main__":
    main()
