# paired t test for BAC statisctic
import glob
import os

import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate

datasets_names = []

# dir_name = 'results_np'
dir_name = 'weighted_results_np'
os.chdir(dir_name)
for file in glob.glob("*.npy"):
    datasets_names.append(file)

os.chdir("..")

# metric_id = 0 #recall
# metric_id = 1 #precision
# metric_id = 2 #specificity
# metric_id = 3  # f1_score
# metric_id = 4 #g_mean
metric_id = 5 #bac

headers = ["n=100 C=2^-5", "n=100 C=2^-2", "n=100 C=2^0", "n=100 C=2^2", "n=100 C=2^5",
           "n=500 C=2^-5", "n=500 C=2^-2", "n=500 C=2^0", "n=500 C=2^2", "n=500 C=2^5",
           "n=1000 C=2^-5", "n=1000 C=2^-2", "n=1000 C=2^0", "n=1000 C=2^2", "n=1000 C=2^5"]

names_column = np.array([["n=100 C=2^-5"], ["n=100 C=2^-2"], ["n=100 C=2^0"], ["n=100 C=2^2"], ["n=100 C=2^5"],
                         ["n=500 C=2^-5"], ["n=500 C=2^-2"], ["n=500 C=2^0"], ["n=500 C=2^2"], ["n=500 C=2^5"],
                         ["n=1000 C=2^-5"], ["n=1000 C=2^-2"], ["n=1000 C=2^0"], ["n=1000 C=2^2"], ["n=1000 C=2^5"]])

ranks_all_datasets = []
all_mean_results = []
for dataset_id, name in enumerate(datasets_names):
    # print(f"dataset name:{name.replace('resultsOf_', '').replace('.npy', '')}")

    results = np.load(f'{dir_name}/{name}')
    results = results[:, :, metric_id]

    mean_results = np.mean(results, axis=1)
    all_mean_results.append(mean_results.tolist())
    ranks_all_datasets.append(rankdata(mean_results).tolist())

ranks_all_datasets = np.array(ranks_all_datasets)
# print(f"Ranks among all datases: {ranks_all_datasets}")
mean_ranks = np.mean(ranks_all_datasets, axis=0)
print(f"Mean ranks among all datasets: {mean_ranks}")
mean_all_mean_results = np.mean(np.array(all_mean_results), axis=0)
# print(f"Mean results of all datasets: {mean_all_mean_results}")

alfa = .05
w_statistic = np.zeros((15, 15))
p_value = np.zeros((15, 15))

for i in range(15):
    for j in range(15):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_all_datasets.T[i], ranks_all_datasets.T[j])

w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
# print(w_statistic_table)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((15, 15))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
# print("\nAdvantage:\n", advantage_table)

significance = np.zeros((15, 15))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
# print("\nStatistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table, "\n\n")
