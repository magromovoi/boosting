from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        i_bs = np.random.choice(x.shape[0], size=int(self.subsample * x.shape[0]), replace=True)
        x_bs = x[i_bs]
        y_bs = y[i_bs]
        p_bs = predictions[i_bs]
        s = -self.loss_derivative(y_bs, p_bs)
        new_base_model = self.base_model_class(**self.base_model_params).fit(x_bs, s)
        new_predictions = new_base_model.predict(x)
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        self.gammas.append(gamma)
        self.models.append(new_base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['val_loss'].append(self.loss_fn(y_valid, valid_predictions))
            if self.early_stopping_rounds is not None:
                val_loss = self.loss_fn(y_valid, valid_predictions)
                if np.all(self.validation_loss < val_loss):
                    break
                else:
                    if self.early_stopping_rounds == 1:
                        self.validation_loss = np.array([val_loss])
                    elif self.early_stopping_rounds > 1:
                        self.validation_loss = np.append(self.validation_loss[1:], val_loss)

        if self.plot:
            sns.set(font_scale=1.)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.arange(self.n_estimators), self.history['train_loss'], label='Train Loss')
            ax.plot(np.arange(self.n_estimators), self.history['val_loss'], label='Val Loss')
            ax.set_xlabel('Iteration')
            ax.set_title('Gradient Boosting Loss by Iteration')
            ax.legend()
            plt.show()
        return self

    def predict_proba(self, x):
        pred = 0
        for gamma, model in zip(self.gammas, self.models):
            pred += self.learning_rate * gamma * model.predict(x)
        proba = self.sigmoid(pred)
        return np.array([1 - proba, proba]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self) -> np.array:
        f_i = np.mean(np.array([model.feature_importances_ for model in self.models]), axis=0)
        return f_i / np.sum(f_i)
