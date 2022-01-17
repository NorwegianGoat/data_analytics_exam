import pandas as pd
import os

__DATA_PATH = './ml-25m'


def load_data(path: str) -> pd.DataFrame:
    movies = pd.read_csv(os.path.join(
        __DATA_PATH, "movies.csv"), index_col="movieId", usecols=["movieId", "genres"])
    ratings = pd.read_csv(os.path.join(__DATA_PATH, "ratings.csv"))
    # Calc. avg. rating foreach movie
    y = ratings.groupby("movieId")["rating"].mean()
    movies = movies.merge(y, on="movieId")
    # One hot encoding for pipe separated genres
    genres = movies["genres"].str.get_dummies()
    movies = movies.merge(genres, on="movieId")
    # Remove old genres column
    movies.drop("genres", axis=1, inplace=True)
    return movies


def bin_rating(ratings: pd.Series) -> pd.Series:
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    binned_ratings = pd.cut(
        ratings, bins=bins, labels=bins[1:])
    return binned_ratings


if __name__ == "__main__":
    df = load_data(__DATA_PATH)
    # df.info(show_counts=True)
    # print(df.head())
    # Binning avg rating
    df["rating"] = bin_rating(df["rating"])
