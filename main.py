import io
from typing import Tuple
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB


__DATA_PATH = './ml-25m'
__IMG_PATH = './img'
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


def dim_reduction(X_train, X_val, X_test):
    vtm = 0.8
    logging.info("Dimensionality reduction. " + str(vtm) +
                 "% of the variance will be mantained")
    logging.debug("Shape before dim. reduction" + str(X_train.shape))
    pca = PCA(random_state=__SEED)
    pca.fit(X_train)
    exp_variance = pca.explained_variance_ratio_
    i, var = 0, 0
    for variance in exp_variance:
        # The components that explain the vtm% of the variance
        if var >= vtm:
            pca = PCA(n_components=i, random_state=__SEED)
            break
        i += 1
        var += variance
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    figure = plt.plot(pca.explained_variance_ratio_)
    plot(figure, ["Eigenvector", "Explained var."], "pca_variance")
    logging.debug("Shape after dim. reduction" + str(X_train.shape))
    return X_train, X_val, X_test


def resample_data(X_train, y_train, encoder) -> Tuple[pd.DataFrame, pd.Series]:
    logging.info("Resampling data")
    figure = plt.hist(encoder.inverse_transform(y_train), bins="auto")
    xy_labels = ["Class", "Freq."]
    plot(figure, xy_labels, "bef_resample")
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.debug("Data before resampling: " + str(count))
    ros = RandomOverSampler(random_state=__SEED)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.debug("Data after the sampling" + str(count))
    figure = plt.hist(encoder.inverse_transform(y_train), bins="auto")
    plot(figure, xy_labels, "aft_resample")
    return X_train, y_train


def analyze_data(X_train: pd.DataFrame, Y_train: pd.Series, x_test: pd.DataFrame, y_test):
    # TODO: Test standardized and normalized data
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, Y_train)
    y_predict = nb_classifier.predict(x_test)
    print("Avg. accuracy nb:", nb_classifier.score(x_test, y_test))


def plot(figure, axis_labels, fig_name):
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.savefig(os.path.join(__IMG_PATH, fig_name))
    plt.clf()


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
    # Dimensionality reduction
    X_train, X_val, X_test = dim_reduction(X_train, X_val, X_test)
    # Data resampling
    X_train, y_train = resample_data(X_train, y_train, encoder)
    # Analysis
    analyze_data(X_train, y_train, X_test, y_test)
