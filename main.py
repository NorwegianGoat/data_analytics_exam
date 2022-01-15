from calendar import day_abbr
import pandas
import os

__DATA_PATH = './ml-25m'


def load_data(path: str) -> pandas.DataFrame:
    files = [os.path.join(__DATA_PATH, file) for file in os.listdir(path)]
    files.remove('./ml-25m/README.txt')
    print('Scanning', __DATA_PATH, "found:", files)
    return [pandas.read_csv(file) for file in files]


if __name__ == "__main__":
    dfs = load_data(__DATA_PATH)
