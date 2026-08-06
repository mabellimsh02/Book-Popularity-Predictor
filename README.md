# Book Popularity Predictor

## What this project does

We want to guess whether a book will be "popular" - meaning lots of people
read and rated it - just from information you'd see on its listing page:
its title, its description, its genre, its average rating, how many pages
it has, and how many other books its author has written.

There's no "popularity" column in the raw data, so we build one ourselves:
a book counts as popular if it's in the top 25% by number of ratings. Then
we train two models to predict that label from everything *except* the
ratings count itself (since that's literally what defines the label), and
see which one does better and what it actually learned to pay attention to.

Dataset: ~5.2k books from a public mirror of Kaggle's "7k Books with
Metadata" (dylanjcastillo/7k-books-with-metadata), mixing text (title,
description) with numeric/categorical metadata (genre, average rating,
page count, publication year).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python get_data.py             # one-time download into data/books.csv
python popularity_predictor.py
```

Prints dtype/missing-value inspection, class balance, and a classification
report for both models, and saves four plots:

- `rating_distribution.png` - average rating and log-scaled ratings-count
  distributions
- `category_breakdown.png` - book counts for the top 15 genres
- `confusion_matrix.png` - tuned Random Forest predictions vs. actuals
- `feature_importance.png` - top 20 features driving the Random Forest

## Project Structure

```
Book-Popularity-Predictor/
├── get_data.py              # Downloads the dataset into data/books.csv
├── popularity_predictor.py  # Builds the label, trains/evaluates both models, and saves plots
├── requirements.txt         # Python dependencies
├── .gitignore
└── data/books.csv           # Created by get_data.py (not committed to the repo)
```

## Defining "popularity"

There's no popularity label in the raw data, so it's derived: a book is
**popular** if its `ratings_count` falls in the top quartile (≥ ~6,600
ratings in this dataset), and not popular otherwise. Ratings count is used
instead of average rating because it proxies reach (how many readers a book
found) rather than reception (whether readers liked it) - a niche book can
average 4.8 stars from 40 ratings and still not be "popular" in any
meaningful sense.

Because the label is built directly from `ratings_count`, that column is
excluded from the feature set - including it would leak the target back in
as a predictor and the model would trivially learn the threshold instead of
anything real about the book.

## Features

- **Numeric**: `average_rating`, `num_pages`, `book_age` (2026 -
  published_year), `description_length`, `title_length`,
  `author_book_count` (how many other books by the same author appear in
  the dataset - a proxy for an author's track record)
- **Categorical**: `category_grouped` - the genre, with everything outside
  the 15 most common categories bucketed into "Other" (479 raw genre values
  is too sparse to one-hot directly)
- **Text**: TF-IDF (unigrams + bigrams, top 300 terms) over the title and
  description concatenated together

All three groups are combined with a `ColumnTransformer` so scaling,
one-hot encoding, and TF-IDF happen in one fitted pipeline - no leakage
between train/test folds.

## Models

1. **Baseline** - Logistic Regression, `class_weight="balanced"` (popular
   books are ~25% of the data).
2. **Tuned** - Random Forest, hyperparameters (`n_estimators`, `max_depth`,
   `min_samples_leaf`, `max_features`) selected via `RandomizedSearchCV`
   with 5-fold stratified CV, scoring on F1 for the popular class.

## Results

The Random Forest improves overall accuracy over the logistic baseline
(0.71 vs. 0.67) and trades recall for precision on the popular class
(precision 0.44 vs. 0.39, recall 0.53 vs. 0.60) - it ends up roughly a wash
on F1, ~0.48 for both. This is a genuinely hard prediction problem: whether
a book "catches on" with tens of thousands of readers depends heavily on
marketing, timing, and word of mouth that no metadata field captures.

What the feature importances do show: an author's existing book count is
the single strongest signal (established authors have built-in audiences),
followed by page count and average rating, then being in the Fiction genre.
Text features contribute individually small but collectively meaningful
signal - words like "war", "battle", and "love" showing up among the top 20
suggests genre-adjacent themes correlate with reach even after accounting
for the genre label itself.
