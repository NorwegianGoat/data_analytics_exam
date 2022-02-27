import io
from time import time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Subset

__DATA_PATH = './ml-25m'
__IMG_PATH = './img'
__SEED = 42
__logging_level = logging.INFO


def _available_devices() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, Y: pd.Series):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.classes = len(self.Y.unique())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, id):
        return self.X[id, :], self.Y[id]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size: int, number_hidden_layers: int):
        super(NeuralNetwork, self).__init__()
        self.device = _available_devices()
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.number_hidden_layers = number_hidden_layers
        # Input layer + output layer + hidden layers
        self.number_of_layers = number_hidden_layers+2
        self.output_layer_size = output_layer_size  # The number of output classes
        self.activation_function = torch.nn.ReLU()
        # Building the layers
        layers = OrderedDict()
        layers[str(0)] = torch.nn.Linear(input_layer_size, hidden_layer_size)
        layers[str(1)] = self.activation_function
        for i in range(0, number_hidden_layers):
            layers[str(len(layers))] = torch.nn.Linear(
                hidden_layer_size, hidden_layer_size)
            layers[str(len(layers))] = self.activation_function
        layers[str(len(layers))] = torch.nn.Linear(
            hidden_layer_size, output_layer_size)
        self.linear_relu_stack = torch.nn.Sequential(layers)
        self.to(self.device)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def _train(self, criterion, optimizer, epochs, data: DataLoader):
        self.train()
        loss_over_epochs = []
        for epoch in range(epochs):
            # Minibatch, we iterate over data passed by data loader for each epoch
            epoch_loss = []
            for batch in data:
                # We only send the the batches we use to device, in this way we
                # use less GPU ram
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                # Resets gradient, otherwise it's summed over time
                optimizer.zero_grad()
                # Foward
                y_pred = self.forward(x)
                # Backpropagation
                loss = criterion(y_pred, y)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            loss = 0
            for loss_i in epoch_loss:
                loss += loss_i
            loss = loss/data.batch_size
            logging.info("Epoch: " + str(epoch) + " avg. loss: " +
                         str(loss))
            loss_over_epochs.append(loss)
        # Returns the trained neural net and the loss over the training
        return self, loss_over_epochs

    def _test(self, X_val, Y_val):
        # Puts the nn in test mode, no dropout, etc.
        self.eval()
        y_pred = self.forward(X_val)
        logging.info(y_pred)
        y_pred = y_pred.argmax()

    def __str__(self) -> str:
        return str({"input_size": self.input_layer_size,
                    "hidden_size": self.hidden_layer_size,
                    "number_of_layers": self.number_of_layers,
                    "number_hidden_layers": self.number_hidden_layers,
                    "output_layer_size": self.output_layer_size,
                    "activation_function": self.activation_function})


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
    n_components = 0.8
    logging.info(
        "Dimensionality reduction. " + str(n_components*100) + "% of the variance will be mantained")
    logging.debug("Shape before dim. reduction" + str(X_train.shape))
    pca = PCA(n_components, random_state=__SEED)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    # plt.plot(pca.explained_variance_ratio_)
    # plot(["Eigenvector", "Explained var."], "pca_variance")
    logging.debug("Shape after dim. reduction" + str(X_train.shape))
    return X_train, X_val, X_test


def resample_data(X_train, y_train, encoder) -> Tuple[pd.DataFrame, pd.Series]:
    logging.info("Resampling data")
    plt.hist(encoder.inverse_transform(y_train), bins="auto")
    xy_labels = ["Class", "Freq."]
    plot(xy_labels, "bef_resample")
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.debug("Data before resampling: " + str(count))
    ros = RandomOverSampler(random_state=__SEED)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    count = np.unique(encoder.inverse_transform(y_train), return_counts=True)
    logging.debug("Data after the sampling" + str(count))
    plt.hist(encoder.inverse_transform(y_train), bins="auto")
    plot(xy_labels, "aft_resample")
    return X_train, y_train


def analyze_data(X_train: pd.DataFrame, Y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    # TODO: Test standardized and normalized data
    # Naive Bayes
    '''nb = GaussianNB()
    nb.fit(X_train, Y_train)
    nb.predict(x_test)
    print("NB accuracy on training",
          nb.score(X_train, Y_train))
    print("NB accuracy on testing:", nb.score(x_test, y_test))
    # Random forest classifier
    rf = RandomForestClassifier(random_state=__SEED)
    rf.fit(X_train, Y_train)
    rf.predict(x_test)
    print("RF accuracy on training: " + str(rf.score(X_train, Y_train)))
    print("RF accuracy on testing: " + str(rf.score(x_test, y_test)))
    # SVM
    svc = SVC(kernel='rbf')
    svc.fit(X_train, Y_train)
    svc.predict(x_test)
    print("SVC accuracy on training:", svc.score(X_train, Y_train))
    print("SVC accuracy on testing:", svc.score(x_test, y_test))'''
    # MLP
    logging.info("This device has " +
                 _available_devices().type + " available.")
    # MLP hyperparams
    hidden_layer_size = 512
    number_hidden_layers = 2
    lr = 0.01
    momentum = 0.09
    batch_size = 32
    epochs = 100
    train_loader = DataLoader(
        Dataset(X_train, Y_train), batch_size, shuffle=True, drop_last=True)
    mlp = NeuralNetwork(X_train.shape[1], hidden_layer_size, len(
        Y_train.unique()), number_hidden_layers)
    logging.info(mlp)
    mlp, loss = mlp._train(torch.nn.CrossEntropyLoss(), torch.optim.SGD(
        mlp.parameters(), lr, momentum), epochs, train_loader)
    plt.plot(range(epochs), loss)
    plot(["Epochs", "Loss"], "mlp_loss_progr")
    # test_loader = DataLoader(Dataset(x_test, y_test),batch_size, shuffle=True, drop_last=True)
    # TODO: Test neural network


def plot(axis_labels, fig_name):
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.savefig(os.path.join(__IMG_PATH, fig_name))
    plt.clf()


if __name__ == "__main__":
    t0 = time()
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
    logging.info("Elapsed time " + str(time()-t0) + "s")
