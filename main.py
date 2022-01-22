import io
from typing import Tuple
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


__DATA_PATH = './ml-25m'
__SEED = 42
__logging_level = logging.INFO


def load_data(path: str) -> pd.DataFrame:
    logging.info("Loading data from " + path)
    movies = pd.read_csv(os.path.join(
        path, "movies.csv"), index_col="movieId", usecols=["movieId", "genres"])
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    genome_scores = pd.read_csv(os.path.join(path, "genome-scores.csv"))
    genome_tags = pd.read_csv(os.path.join(path, "genome-tags.csv"))
    # Calc. avg. rating foreach movie
    y = ratings.groupby("movieId")["rating"].mean()
    movies = movies.merge(y, on="movieId")
    # One hot encoding for pipe separated genres
    genres = movies["genres"].str.get_dummies()
    movies = movies.merge(genres, on="movieId")
    # Remove old genres column
    movies.drop("genres", axis=1, inplace=True)
    # Adding genome
    genome_scores = genome_scores.merge(genome_tags, on="tagId")
    genome_scores = genome_scores.pivot(
        index="movieId", columns="tag", values="relevance")
    movies = movies.merge(genome_scores, on="movieId")
    return movies


def rating_discretization(ratings: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    binned_ratings = pd.cut(
        ratings, bins=bins, labels=bins[1:])
    label_encoder = LabelEncoder()
    label_encoder.fit(binned_ratings)
    binned_ratings = label_encoder.transform(binned_ratings)
    return binned_ratings, label_encoder


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    # Avg rating discretization
    logging.info("Discretizing data in categories")
    bins, encoder = rating_discretization(df["rating"])
    df["rating"] = bins
    df_info = io.StringIO()
    df.info(show_counts=True, buf=df_info)
    logging.debug(df)
    logging.debug(df_info.getvalue())
    logging.info("NA values: " + str(df.isna().sum().sum()))
    return df, encoder


def resample_data(X_train, y_train) -> Tuple[pd.DataFrame, pd.Series]:
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.info("Data before resampling: " + str(count))
    ros = RandomOverSampler(random_state=__SEED)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.info("Data after the sampling" + str(count))
    return X_train, y_train


def analyze_data(X_train: pd.DataFrame, Y_train: pd.Series, x_test: pd.DataFrame, y_test):
    # TODO: Test standardized and normalized data
    pass


def plot(df: pd.DataFrame):
    plt.hist(df["rating"], bins="auto")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=__logging_level)
    # Load data
    df = load_data(__DATA_PATH)
    # Data pre-processing
    df, encoder = preprocess_data(df)
    # Train, Validation, Test split
    logging.info("Splitting data in train, validation, test")
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != "rating"], df["rating"], test_size=0.2,
        random_state=__SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=__SEED)
    # Data resampling
    X_train, y_train = resample_data(X_train, y_train)
    # Analysis
    analyze_data(X_train, y_train, X_test, y_test)
    # Plot
    plot(df)
