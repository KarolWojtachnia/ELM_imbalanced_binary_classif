import glob
import os
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from elm import ExtremeLearningMachine as elm
import numpy as np

ELMs = {
    "ELM_100_2^-5": elm(pow(2, -5), True, 100),
    "ELM_100_2^-2": elm(pow(2, -2), True, 100),
    "ELM_100_2^0": elm(pow(2, 0), True, 100),
    "ELM_100_2^2": elm(pow(2, 2), True, 100),
    "ELM_100_2^5": elm(pow(2, 5), True, 100),
    "ELM_500_2^-5": elm(pow(2, -5), True, 500),
    "ELM_500_2^-2": elm(pow(2, -2), True, 500),
    "ELM_500_2^0": elm(pow(2, 0), True, 500),
    "ELM_500_2^2": elm(pow(2, 2), True, 500),
    "ELM_500_2^5": elm(pow(2, 5), True, 500),
    "ELM_1000_2^-5": elm(pow(2, -5), True, 1000),
    "ELM_1000_2^-2": elm(pow(2, -2), True, 1000),
    "ELM_1000_2^0": elm(pow(2, 0), True, 1000),
    "ELM_1000_2^2": elm(pow(2, 2), True, 1000),
    "ELM_1000_2^5": elm(pow(2, 5), True, 1000)
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
os.chdir("datasets")
for file in glob.glob("*.csv"):
    datasets_names.append(file)
os.chdir("..")

n_splits = 2
n_repeats = 5

for i in range(len(datasets_names)):

    dataset_name = datasets_names[i]
    dataset = np.genfromtxt("datasets\%s" % dataset_name, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    y[y==0] = -1

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=58)

    scores = np.zeros((len(ELMs), n_splits * n_repeats, len(metrics)))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for ELM_id, ELM in enumerate(ELMs):
            clf = clone(ELMs[ELM])

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            y_true = np.copy(y)
            y_true[y_true == -1] = 0
            y_pred[y_pred == -1] = 0

            for metric_id, metric in enumerate(metrics):
                scores[ELM_id, fold_id, metric_id] = metrics[metric](y_true[test], y_pred)

    scores[np.isnan(scores)] = 0
    np.save(f'weighted_results_np/results_{dataset_name.replace(".csv", "")}', scores)
    mean_scores = np.mean(scores, axis=1)

    np.savetxt(f'weighted_mean_results/results_{dataset_name}', mean_scores, delimiter=",")
