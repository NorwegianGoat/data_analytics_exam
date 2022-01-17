import io
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

__DATA_PATH = './ml-25m'
__SEED = 42
__logging_level = logging.DEBUG


def load_data(path: str) -> pd.DataFrame:
    movies = pd.read_csv(os.path.join(
        __DATA_PATH, "movies.csv"), index_col="movieId", usecols=["movieId", "genres"])
    ratings = pd.read_csv(os.path.join(__DATA_PATH, "ratings.csv"))
    genome = pd.read_csv(os.path.join(__DATA_PATH, "genome-scores.csv"))
    # Calc. avg. rating foreach movie
    y = ratings.groupby("movieId")["rating"].mean()
    movies = movies.merge(y, on="movieId")
    # One hot encoding for pipe separated genres
    genres = movies["genres"].str.get_dummies()
    movies = movies.merge(genres, on="movieId")
    # Remove old genres column
    movies.drop("genres", axis=1, inplace=True)
    # Adding genome
    genome = genome.pivot(index="movieId", columns="tagId", values="relevance")
    movies = movies.merge(genome, on="movieId")
    return movies


def bin_rating(ratings: pd.Series) -> pd.Series:
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    binned_ratings = pd.cut(
        ratings, bins=bins, labels=bins[1:])
    return binned_ratings


def plot(df: pd.DataFrame):
    plt.hist(df["rating"], bins="auto")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=__logging_level)
    # Load data
    df = load_data(__DATA_PATH)
    # Binning avg rating
    df["rating"] = bin_rating(df["rating"])
    df_info = io.StringIO()
    df.info(show_counts=True, buf=df_info)
    logging.info(df_info.getvalue())
    logging.info("NA values: " + str(df.isna().sum().sum()))
    logging.debug(df)
    logging.debug(df.loc[:, df.columns != "rating"])
    # Dimensionality reduction
    # Train, Validation, Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != "rating"], df["rating"], test_size=0.2,
        random_state=__SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=__SEED)
    # Plot
    # plot(df)
