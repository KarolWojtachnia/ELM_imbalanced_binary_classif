from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from elm import ExtremeLearningMachine as elm
import numpy as np

dataset = 'datasets\ecoli4.csv'
# dataset = 'datasets\glass4.csv'
dataset = np.genfromtxt(dataset, delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
for i in range(y.shape[0]):
    if y[i] == 0:
        y[i] = -1

C_values = {
    "2^-5": pow(2, -5),
    "2^-2": pow(2, -2),
    "2^0": pow(2, 0),
    "2^2": pow(2, 2),
    "2^5": pow(2, 5)
}

hidden_nodes = {
    "100": 100,
    "1000": 1000,
    "10000": 10000
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=58)

scores = np.zeros(len(C_values), len(hidden_nodes), (n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for C_value_id, C_value in enumerate(C_values):
        for hidden_id, hidden in enumerate(hidden_nodes):
            clf = elm(C_value, hidden)
            X_train, y_train = X[train], y[train]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])

            for metric_id, metric in enumerate(metrics):
                scores[C_value_id, hidden_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)

np.save('ELMresults', scores)
