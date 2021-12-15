import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import glob
import os
from strlearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from elm import ExtremeLearningMachine as elm
import numpy as np

models = {
    "ELM": elm(pow(2, 5), False, 1000),
    "ELM_W": elm(pow(2, 5), True, 1000),
    "MLP": MLPClassifier(),
    "SVC": SVC(),
    "GNB": GaussianNB()
}

datasets_names = []
os.chdir("datasets")
for file in glob.glob("*.csv"):
    datasets_names.append(file)
os.chdir("..")

n_splits = 2
n_repeats = 5

all_results = np.zeros((len(datasets_names), len(models), n_splits * n_repeats))

for i in range(len(datasets_names)):

    dataset_name = datasets_names[i]
    dataset = np.genfromtxt("datasets\%s" % dataset_name, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    y[y == 0] = -1

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=58)

    scores = np.zeros((len(models), n_splits * n_repeats))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for model_id, model in enumerate(models):
            clf = clone(models[model])

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            y_true = np.copy(y)
            y_true[y_true == -1] = 0
            y_pred[y_pred == -1] = 0

            scores[model_id, fold_id] = balanced_accuracy_score(y_true[test], y_pred)

    scores[np.isnan(scores)] = 0

    all_results[i] = scores

    np.save(f'other_Classifiers_results/results_{dataset_name.replace(".csv", "")}', scores)
np.save(f'other_Classifiers_results/all_results', all_results)
