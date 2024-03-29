# @Author: Riccardo Mioli - riccardo.mioli2@studio.unibo.it - 983525
# @Date: 01/04/2022 (gg/mm/aaaa)
from joblib import dump, load
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
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import BasicVariantGenerator
import ray

__DATA_PATH = './ml-25m'
__DUMP_MODELS_PATH = './models'
__IMG_PATH = os.path.join(os.path.realpath('.'), 'img')
__SEED = 42
__logging_level = logging.INFO


def _available_devices() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.classes = Y.max().astype(int)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, id):
        return self.X[id, :], self.Y[id]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size: int, number_hidden_layers: int, dropout_prob: int):
        super(NeuralNetwork, self).__init__()
        self.device = _available_devices()
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.number_hidden_layers = number_hidden_layers
        self.dropout_prob = dropout_prob
        # Input layer + output layer + hidden layers
        self.number_of_layers = number_hidden_layers+2
        self.output_layer_size = output_layer_size  # The number of output classes
        self.activation_function = torch.nn.ReLU()
        # Building the layers
        layers = OrderedDict()
        layers[str(len(layers))] = torch.nn.Dropout(p=dropout_prob)
        layers[str(len(layers))] = torch.nn.Linear(
            input_layer_size, hidden_layer_size)
        layers[str(len(layers))] = torch.nn.BatchNorm1d(hidden_layer_size)
        layers[str(len(layers))] = self.activation_function
        for i in range(0, number_hidden_layers):
            layers[str(len(layers))] = torch.nn.Dropout(p=dropout_prob)
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

    def _train(self, criterion, optimizer, epochs, train: DataLoader, validation: Dataset):
        loss_updates = []
        grace = 5  # Used as relaxation condition for early stopping
        for epoch in range(epochs):
            self.train()
            # Minibatch, we iterate over data passed by data loader for each epoch
            for batch in train:
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
                loss_updates.append(loss.item())
                if tune.is_session_enabled():
                    tune.report(loss=loss.item(), accuracy=accuracy_score(
                        y.to("cpu"), y_pred.argmax(dim=1, keepdim=True).squeeze().to("cpu")))
                loss.backward()
                optimizer.step()
            logger.debug("Epoch: " + str(epoch) + " latest loss: " +
                         str(loss_updates[-1]))
            if self._test(validation, criterion) > loss_updates[-1]:
                # If the loss on the validation set is bigger we stop
                # the learning phase
                if grace > 0:
                    grace -= 1
                else:
                    break
        # Returns the trained neural net and the loss over the training
        return self, loss_updates

    def _test(self, testset: DataLoader, criterion=None):
        # Puts the nn in test mode, no dropout, etc.
        self.eval()
        y_test = None
        y_pred = None
        for x, y in testset:
            y_test = y.to(self.device)
            y_pred = self.forward(x.to(self.device))
        if criterion:
            loss = criterion(y_pred, y_test)
            logger.debug("Loss on the valset: %f." % loss.item())
            logger.debug(y_pred)
            return loss.item()
        else:
            return y_pred.argmax(dim=1, keepdim=True).squeeze().to("cpu")

    def __str__(self) -> str:
        return str({"input_size": self.input_layer_size,
                    "hidden_size": self.hidden_layer_size,
                    "number_of_layers": self.number_of_layers,
                    "number_hidden_layers": self.number_hidden_layers,
                    "dropout_probability": self.dropout_prob,
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
    # plot(["Dimension", "Explained var."], "lda_variance")
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


def train_models():
    logger.info("Training models")
    # Params for hyperparams tuner
    n_jobs = os.cpu_count()-1
    n_iter = 10  # 10
    cv = 5  # 5
    verbose = 1  # 1
    scoring = "accuracy"
    btm = {}  # best trained models
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(Xr_train, yr_train)
    y_pred = nb.predict(Xr_test)
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
    rf.fit(Xr_train, yr_train)
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
    svc.fit(Xr_train, yr_train)
    btm['support_vector'] = svc.best_estimator_
    dump(svc.best_estimator_, os.path.join(
        __DUMP_MODELS_PATH, 'support_vector.joblib'))
    logger.info("Support vector best params: " + str(svc.best_params_))
    print("Support vector accuracy score: %f" % svc.best_score_)
    # MLP
    logger.info("This device has " +
                _available_devices().type + " available.")
    tune_res = {'gpu': 1 if _available_devices().type != 'cpu' else 0}
    hidden_layer_size = tune.sample_from(lambda _: 2**np.random.randint(1, 4))
    number_hidden_layers = tune.sample_from(
        lambda _: 2**np.random.randint(1, 4))
    learning_rate = tune.uniform(1e-3, 1e-1)
    momentum = tune.uniform(9e-3, 9e-1)
    batch_size = tune.choice([32, 256, 512, 1024, 2048])
    epochs = tune.choice([100, 200, 300, 400])
    dropout_prob = tune.uniform(0.05, 0.01)
    configs = {"hidden_layer_size": hidden_layer_size, "number_hidden_layers": number_hidden_layers,
               "learning_rate": learning_rate, "momentum": momentum, "batch_size": batch_size,
               "epochs": epochs, "dropout_prob": dropout_prob}

    def train_nn(config: dict, X_train, y_train):
        train_loader = DataLoader(
            Dataset(X_train, y_train), config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(Dataset(X_val, y_val), X_val.shape[0])
        mlp = NeuralNetwork(X_train.shape[1], config['hidden_layer_size'], y_train.max(
        ).astype(int)+1, config['number_hidden_layers'], config['dropout_prob'])
        logger.info(mlp)
        optimizer = torch.optim.SGD(
            mlp.parameters(), config['learning_rate'], config['momentum'])
        mlp, loss = mlp._train(torch.nn.CrossEntropyLoss(
        ), optimizer, config['epochs'], train_loader, val_loader)
        # Local plot (just for this specific training session)
        plt.plot(range(0, len(loss)), loss)
        plot(["Updates", "Loss"], "mlp_loss_progr_minib_bnorm_drop")
        return mlp

    ray.init(log_to_driver=False, logging_level=logging.CRITICAL)
    results = tune.run(tune.with_parameters(train_nn, X_train=X_train, y_train=y_train), config=configs,
                       scheduler=ASHAScheduler(metric="accuracy", mode="max"),
                       local_dir=os.path.realpath("."), verbose=0,
                       search_alg=BasicVariantGenerator(random_state=np.random.RandomState(__SEED)), num_samples=50, resources_per_trial=tune_res)
    # Global plot of jobs pruned by asha scheduler <- loss
    draw = None
    for df in results.trial_dataframes.values():
        draw = df.loss.plot(ax=draw)
    plot(['Updates', 'Loss'], 'loss_early_stopping_ASHAScheduler')
    # Global plot of jobs pruned by asha scheduler <- accuracy
    draw = None
    for df in results.trial_dataframes.values():
        draw = df.accuracy.plot(ax=draw)
    plot(['Updates', 'Accuracy'], 'acc_early_stopping_ASHAScheduler')
    # Save best config
    logger.info("Best config nn is: " +
                str(results.get_best_config(metric="accuracy", mode="max")))
    mlp = train_nn(results.get_best_config(
        metric="accuracy", mode="max"), X_train, y_train)
    y_pred = mlp._test(DataLoader(Dataset(X_train, y_train), X_train.shape[0]))
    logger.info("Neural net accuracy: %f." % accuracy_score(y_train, y_pred))
    torch.save(mlp, os.path.join(__DUMP_MODELS_PATH, 'nn_dump'))
    btm['neural_net'] = mlp
    '''# Just for manual test purposes
    btm['neural_net'] = train_nn({"hidden_layer_size": 256, "number_hidden_layers": 2, "learning_rate": 0.04720952643155001,
                                 "momentum": 0.7752069024020616, "batch_size": 512, "epochs": 100, "dropout_prob": 0.017397936280588825}, X_train, y_train)'''
    return btm


def test_models(models: Dict[str, BaseEstimator], Xr_test, X_test, y_test):
    logger.info("Testing models")
    for key, model in models.items():
        if key != "neural_net":
            y_pred = model.predict(Xr_test)
        else:
            testset = DataLoader(Dataset(X_test, y_test), X_test.shape[0])
            y_pred = model._test(testset).numpy().astype(int)
        average = "macro"
        zero_div = 0
        precision = precision_score(
            y_test, y_pred, average=average, zero_division=zero_div)
        recall = recall_score(
            y_test, y_pred, average=average, zero_division=zero_div)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=zero_div)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(key + " Precision: %f. Recall: %f. f1: %f. Accuracy: %f." %
                    (precision, recall, f1, accuracy))
        # Plot confusion
        cm = confusion_matrix(y_test, y_pred, labels=encoder.classes_)
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
    torch.manual_seed(__SEED)
    torch.use_deterministic_algorithms(True)
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
    # Dimensionality reduction. Datasets with r are the ones with dim reduction
    Xr_train, Xr_val, Xr_test = dim_reduction(X_train, X_val, X_test, y_train)
    plt.scatter(Xr_train[:, 0], Xr_train[:, 1], c=y_train)
    plot(["x0", "x1"], "lda_scatterplot_data")
    # Data resampling
    Xr_train, yr_train = resample_data(Xr_train, y_train)
    X_train, y_train = resample_data(X_train, y_train)
    # Models train
    trained_models = train_models()
    test_models(trained_models, Xr_test, X_test, y_test)
    # Models test
    logger.info("Elapsed time " + str(time()-t0) + "s")
