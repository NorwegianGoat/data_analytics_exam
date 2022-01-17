import pandas as pd
import os

__DATA_PATH = './ml-25m'
__genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western", "(no genres listed)"]

def parse_genres(genres: pd.Series) -> pd.DataFrame:
    genres = pd.get_dummies(genres.str.split("|", expand=True))
    print(genres)


def load_data(path: str) -> pd.DataFrame:
    movies = pd.read_csv(os.path.join(__DATA_PATH, "movies.csv"), index_col="movieId")
    ratings = pd.read_csv(os.path.join(__DATA_PATH, "ratings.csv"))
    y = ratings.groupby("movieId")["rating"].mean()
    movies = pd.merge(movies, y, on="movieId")
    #df.info(show_counts=True)
    '''df = df.reindex(axis=1, labels=df.columns.tolist() + __genres)
    for i, genres in df["genres"].items():
        genres = genres.split("|")
        for genre in genres:
            df.iloc[i][genre] = 1
    print(df.head())'''
    return df


if __name__ == "__main__":
    df = load_data(__DATA_PATH)
