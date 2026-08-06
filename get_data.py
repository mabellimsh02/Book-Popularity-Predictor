"""
Overview
--------
This script fetches the book dataset we train on and saves it to disk, so
we don't have to hit the internet every time we run the real project
script.

The data comes from a public copy of Kaggle's "7k Books with Metadata"
dataset: ~5,200 books, each with a title, author, genre, description,
average rating, ratings count, page count, and publication year. That mix
of text + numbers + categories is exactly what we want for this project.

Run this file on its own once (`python get_data.py`), or just run
popularity_predictor.py directly - it calls the same download() function
for you automatically.
"""
import urllib.request
from pathlib import Path

# Where to get the data from, and where to save it once downloaded.
URL = "https://raw.githubusercontent.com/nigelnuique/book-recommender/main/books_cleaned.csv"
DATA_PATH = Path(__file__).parent / "data" / "books.csv"


def download():
    """Save the dataset to data/books.csv, unless we already have it."""
    # Make sure the "data" folder exists before we try to save into it.
    DATA_PATH.parent.mkdir(exist_ok=True)

    # Skip the download if we already downloaded it in a previous run.
    if DATA_PATH.exists():
        print(f"Already have {DATA_PATH}")
        return DATA_PATH

    # Otherwise, fetch the CSV from the URL above and write it to disk.
    print(f"Downloading dataset to {DATA_PATH} ...")
    urllib.request.urlretrieve(URL, DATA_PATH)
    print("Done.")
    return DATA_PATH


if __name__ == "__main__":
    download()
