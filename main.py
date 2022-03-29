from joblib import dump, load
import argparse
import io
from time import time
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection._search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from ray import tune
import seaborn as sn

__DATA_PATH = './ml-25m'
__DUMP_MODELS_PATH = './models'
__IMG_PATH = os.path.join(os.path.realpath('.'), 'img')
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
        layers[str(len(layers))] = torch.nn.Linear(
            input_layer_size, hidden_layer_size)
        layers[str(len(layers))] = torch.nn.BatchNorm1d(hidden_layer_size)
        layers[str(len(layers))] = self.activation_function
        for i in range(0, number_hidden_layers):
            layers[str(len(layers))] = torch.nn.Linear(
                hidden_layer_size, hidden_layer_size)
            layers[str(len(layers))] = torch.nn.BatchNorm1d(hidden_layer_size)
            layers[str(len(layers))] = self.activation_function
        layers[str(len(layers))] = torch.nn.Linear(
            hidden_layer_size, output_layer_size)
        self.linear_relu_stack = torch.nn.Sequential(layers)
        self.to(self.device)

    def forward(self, x):
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
            loss = np.mean(epoch_loss)
            logger.info("Epoch: " + str(epoch) + " avg. loss: " +
                        str(loss))
            loss_over_epochs.append(loss)
        # Returns the trained neural net and the loss over the training
        return self, loss_over_epochs

    def _test(self, X_val, Y_val):
        # Puts the nn in test mode, no dropout, etc.
        self.eval()
        y_pred = self.forward(X_val)
        logger.info(y_pred)
        y_pred = y_pred.argmax()

    def __str__(self) -> str:
        return str({"input_size": self.input_layer_size,
                    "hidden_size": self.hidden_layer_size,
                    "number_of_layers": self.number_of_layers,
                    "number_hidden_layers": self.number_hidden_layers,
                    "output_layer_size": self.output_layer_size,
                    "activation_function": self.activation_function})


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from " + path)
    movies = pd.read_csv(os.path.join(
        path, "movies.csv"), index_col="movieId", usecols=["movieId", "genres"])
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    genome_scores = pd.read_csv(os.path.join(path, "genome-scores.csv"))
    genome_tags = pd.read_csv(os.path.join(path, "genome-tags.csv"))
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
    # Calc. avg. rating foreach movie
    y = ratings.groupby("movieId")["rating"].mean()
    movies = movies.merge(y, on="movieId")
    # Print info on data
    df_info = io.StringIO()
    movies.info(show_counts=True, buf=df_info)
    logger.debug(movies)
    logger.debug(df_info.getvalue())
    logger.info("NA values: %i." % movies.isna().sum().sum())
    # Check data integrity
    df_info = movies.describe().loc[["min", "max"], :]
    condition0 = df_info < 0
    logger.info("Values under 0: %i." % condition0.sum().sum())
    condition1 = df_info.loc[:, df_info.columns != "rating"] > 1
    logger.info("Characteristics over 1: %i." % condition1.sum().sum())
    condition2 = df_info.loc[:, df_info.columns == "rating"] > 5
    logger.info("Ratings over 5: %i." % condition2.sum().sum())
    return movies


def _rating_discretization(ratings: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    binned_ratings = pd.cut(
        ratings, bins=bins, labels=bins[1:])
    label_encoder = LabelEncoder()
    label_encoder.fit(binned_ratings)
    binned_ratings = label_encoder.transform(binned_ratings)
    return binned_ratings, label_encoder


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder, Normalizer]:
    # Avg rating discretization (binning)
    logger.info("Discretizing ratings")
    bins, encoder = _rating_discretization(df['rating'])
    df['rating'] = bins
    logger.debug(df.head)
    # Scaling data
    logger.info("Scaling data")
    scaler = Normalizer(copy=False, norm='l2')
    # scaler = StandardScaler(copy=False)
    # scaler = MinMaxScaler(copy=False)
    df = scaler.fit_transform(df.to_numpy())
    df[:, -1] = bins
    logger.debug(df)
    return df, encoder, scaler


def dim_reduction(X_train, X_val, X_test, y_train):
    logger.debug("Shape before dim. reduction" + str(X_train.shape))
    # n_components = 0.8
    # projector = PCA(n_components)
    projector = LinearDiscriminantAnalysis()
    projector.fit(X_train, y_train)  # y_train is automatically ignored in LDA
    X_train = projector.transform(X_train)
    X_val = projector.transform(X_val)
    X_test = projector.transform(X_test)
    # plt.plot(projector.explained_variance_ratio_)
    #plot(["Dimension", "Explained var."], "lda_variance")
    logger.debug("Shape after dim. reduction" + str(X_train.shape))
    logger.debug(X_train)
    return X_train, X_val, X_test


def resample_data(X_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Resampling data")
    plt.hist(encoder.inverse_transform(y_train.astype(int)), bins="auto")
    xy_labels = ["Class", "Freq."]
    plot(xy_labels, "bef_resample")
    count = np.unique(y_train, return_counts=True)
    logger.debug("Data before resampling: " + str(count))
    # oversampler = RandomOverSampler()
    oversampler = SMOTE(k_neighbors=6)  # 5,6,7,8,4
    # oversampler = BorderlineSMOTE()
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    count = np.unique(y_train, return_counts=True)
    logger.debug("Data after the sampling" + str(count))
    plt.hist(encoder.inverse_transform(y_train.astype(int)), bins="auto")
    plot(xy_labels, "aft_resample")
    return X_train, y_train


def train_models(X_train: np.ndarray, Y_train: np.ndarray, x_test: np.ndarray, y_test):
    logger.info("Training models")
    # Params for hyperparams tuner
    n_jobs = os.cpu_count()-1
    n_iter = 10  # 10
    cv = 5  # 5
    verbose = 1
    scoring = "accuracy"
    btm = {}  # best trained models
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    y_pred = nb.predict(x_test)
    print("Naive Bayes accuracy score %f" %
          accuracy_score(y_test, y_pred))
    dump(nb, os.path.join(__DUMP_MODELS_PATH, 'naive_bayes.joblib'))
    btm['naive_bayes'] = nb
    # Random forest classifier
    hyperparams = {"n_estimators": list(range(100, 350, 50)), 'criterion': ['gini', 'entropy'],
                   'max_depth': list(range(10, 30, 5))+[None], 'min_samples_split': list(range(2, 11, 2))}
    estimator = RandomForestClassifier(n_jobs, random_state=__SEED)
    rf = RandomizedSearchCV(estimator, hyperparams, n_jobs=n_jobs, verbose=verbose,
                            cv=cv, scoring=scoring, n_iter=n_iter)
    rf.fit(X_train, Y_train)
    btm['random_forest'] = rf.best_estimator_
    dump(rf.best_estimator_, os.path.join(
        __DUMP_MODELS_PATH, 'random_forest.joblib'))
    logger.info("Random forest best params: " + str(rf.best_params_))
    print('Random forest accuracy score: %f' % rf.best_score_)
    # SVM
    hyperparams = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                   'degree': list(range(2, 5)), 'tol': np.linspace(1e-3, 1e-5, 5),
                   'C': np.linspace(1, 5, 5)}
    estimator = SVC()
    svc = RandomizedSearchCV(estimator, hyperparams, n_jobs=n_jobs,
                             verbose=verbose, cv=cv, scoring=scoring, n_iter=n_iter)
    svc.fit(X_train, Y_train)
    btm['support_vector'] = svc.best_estimator_
    dump(svc.best_estimator_, os.path.join(
        __DUMP_MODELS_PATH, 'support_vector.joblib'))
    logger.info("Support vector best params: " + str(svc.best_params_))
    print("Support vector accuracy score: %f" % svc.best_score_)
    '''# MLP
    logger.info("This device has " +
                _available_devices().type + " available.")
    # MLP hyperparams
    tune_res = {'gpu': 1 if _available_devices().type != 'cpu' else 0}
    hidden_layer_size = tune.sample_from(lambda _: 2**np.random.randint(3, 10))
    number_hidden_layers = tune.sample_from(
        lambda _: 2**np.random.randint(1, 6))
    learning_rate = tune.loguniform(1e-3, 1e-1)
    momentum = tune.loguniform(9e-3, 9e-2)
    batch_size = tune.choice([8, 16, 32, 64])
    epochs = tune.choice([50, 100, 200, 400])
    configs = {"hidden_layer_size": hidden_layer_size, "number_hidden_layers": number_hidden_layers,
               "learning_rate": learning_rate, "momentum": momentum, "batch_size": batch_size,
               "epochs": epochs}

    def train_nn(config: dict):
        train_loader = DataLoader(
            Dataset(X_train, Y_train), config['batch_size'], shuffle=True, drop_last=True)
        mlp = NeuralNetwork(X_train.shape[1], config['hidden_layer_size'], len(
            Y_train.unique()), config['number_hidden_layers'])
        logger.info(mlp)
        mlp, loss = mlp._train(torch.nn.CrossEntropyLoss(), torch.optim.SGD(
            mlp.parameters(), config['learning_rate'], config['momentum']), config['epochs'], train_loader)
        tune.report(mean_loss=loss[-1])
        plt.plot(range(config['epochs']), loss)
        plot(["Epochs", "Loss"], "mlp_loss_progr")
    results = tune.run(train_nn, config=configs,
                       local_dir=os.path.realpath("."), verbose=verbose, num_samples=n_iter, resources_per_trial=tune_res)
    logger.info("Best config nn is: " +
                str(results.get_best_config(metric="mean_loss", mode="min")))'''
    return btm


def test_models(models: Dict[str, BaseEstimator], X_test, Y_test):
    logger.info("Testing models")
    for key, model in models.items():
        y_pred = model.predict(X_test)
        average = "macro"
        zero_div = 0
        precision = precision_score(
            Y_test, y_pred, average=average, zero_division=zero_div)
        recall = recall_score(
            Y_test, y_pred, average=average, zero_division=zero_div)
        f1 = f1_score(Y_test, y_pred, average=average, zero_division=zero_div)
        accuracy = accuracy_score(Y_test, y_pred)
        logger.info(key + " Precision: %f. Recall: %f. f1: %f. Accuracy: %f." %
                    (precision, recall, f1, accuracy))
        # Plot confusion
        cm = confusion_matrix(Y_test, y_pred, labels=encoder.classes_)
        conf_plot = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=encoder.classes_)
        conf_plot.plot()
        plot([None, None], key+"_confusion_matrix")


def plot(axis_labels, fig_name):
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.savefig(os.path.join(__IMG_PATH, fig_name))
    plt.clf()


if __name__ == "__main__":
    t0 = time()
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=__logging_level)
    np.random.seed(__SEED)
    # Load data
    df = load_data(__DATA_PATH)
    # Data pre-processing
    df, encoder, scaler = preprocess_data(df)
    # Train, Validation, Test split
    logger.info("Splitting data in train, validation, test")
    X_train, X_test, y_train, y_test = train_test_split(
        df[:, :-1], df[:, -1], test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25)
    # Dimensionality reduction
    X_train, X_val, X_test = dim_reduction(X_train, X_val, X_test, y_train)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plot(["x0", "x1"], "lda_scatterplot_data")
    # Data resampling
    X_train, y_train = resample_data(X_train, y_train)
    # Models train
    trained_models = train_models(X_train, y_train, X_val, y_val)
    test_models(trained_models, X_test, y_test)
    # Models test
    logger.info("Elapsed time " + str(time()-t0) + "s")
