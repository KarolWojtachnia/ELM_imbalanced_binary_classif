import glob
import os
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from elm import ExtremeLearningMachine as elm
import numpy as np

ELMs = {
    "ELM_100_2^-5": elm(pow(2, -5), False, 100),
    "ELM_100_2^-2": elm(pow(2, -2), False, 100),
    "ELM_100_2^0": elm(pow(2, 0), False, 100),
    "ELM_100_2^2": elm(pow(2, 2), False, 100),
    "ELM_100_2^5": elm(pow(2, 5), False, 100),
    "ELM_500_2^-5": elm(pow(2, -5), False, 500),
    "ELM_500_2^-2": elm(pow(2, -2), False, 500),
    "ELM_500_2^0": elm(pow(2, 0), False, 500),
    "ELM_500_2^2": elm(pow(2, 2), False, 500),
    "ELM_500_2^5": elm(pow(2, 5), False, 500),
    "ELM_1000_2^-5": elm(pow(2, -5), False, 1000),
    "ELM_1000_2^-2": elm(pow(2, -2), False, 1000),
    "ELM_1000_2^0": elm(pow(2, 0), False, 1000),
    "ELM_1000_2^2": elm(pow(2, 2), False, 1000),
    "ELM_1000_2^5": elm(pow(2, 5), False, 1000)
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

datasets_names = []
counter = 0
os.chdir("datasets")
for file in glob.glob("*.csv"):
    datasets_names.append(file)
os.chdir("..")

n_splits = 5
n_repeats = 2

for i in range(len(datasets_names)):

    dataset_name = datasets_names[i]
    dataset = np.genfromtxt("datasets\%s" % dataset_name, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] = -1

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=58)

    scores = np.zeros((len(ELMs), n_splits * n_repeats, len(metrics)))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for ELM_id, ELM in enumerate(ELMs):
            clf = clone(ELMs[ELM])
            X_train, y_train = X[train], y[train]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])
            for i in range(len(y_pred)):
                if y_pred[i] == -1:
                    y_pred[i] = 0

            for metric_id, metric in enumerate(metrics):
                scores[ELM_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)

    scores[np.isnan(scores)] = 0

    mean_scores1 = np.mean(scores, axis=1)

    np.savetxt('results/resultsOf_%s' % dataset_name, mean_scores1, delimiter=",")
