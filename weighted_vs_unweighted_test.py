# paired t test for BAC statisctic
import glob
import os

import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel

datasets_names = []

dir_unweighted = 'results_np'
dir_weighted = 'weighted_results_np'
os.chdir(dir_unweighted)
for file in glob.glob("*.npy"):
    datasets_names.append(file)

os.chdir("..")

# metric_id = 0 #recall
# metric_id = 1 #precision
# metric_id = 2 #specificity
# metric_id = 3  # f1_score
# metric_id = 4 #g_mean
metric_id = 5 #bac

headers = ["no W", "W"]

names_column = np.array([["ecoli-0-1-3-7_vs_2-6"], ["ecoli-0-1-4-6_vs_5"], ["ecoli-0-1_vs_2-3-5"], ["ecoli-0-2-3-4_vs_5"], ["ecoli-0-3-4-7_vs_5-6"],
                        ["ecoli-0-6-7_vs_3-5"], ["ecoli-0-6-7_vs_5"], ["ecoli4"], ["glass-0-1-6_vs_2"], ["glass-0-1-6_vs_5"],
                        ["glass-0-4_vs_5"], ["glass4"], ["glass5"], ["page-blocks-1-3_vs_4"], ["shuttle-c0-vs-c4"],
                        ["shuttle-c2-vs-c4"], ["yeast-0-2-5-6_vs_3-7-8-9"], ["yeast-1-4-5-8_vs_7"], ["yeast-1_vs_7"], ["yeast-2_vs_4"]])

aggregated_results = np.zeros((20, 2, 10))

for dataset_id, name in enumerate(datasets_names):

    results_unweighted = np.load(f'{dir_unweighted}/{name}')
    results_unweighted = results_unweighted[14, :, metric_id]

    results_weighted = np.load(f'{dir_weighted}/{name}')
    results_weighted = results_weighted[14, :, metric_id]

    aggregated_results[dataset_id, 0] = results_unweighted
    aggregated_results[dataset_id, 1] = results_weighted

print(aggregated_results.mean(axis=2))
alfa = .05
t_statistic = np.zeros((20, 2))
p_value = np.zeros((20, 2))

for i in range(20):
    t_statistic[i, 0], p_value[i, 0] = ttest_rel(aggregated_results[i][0], aggregated_results[i][1])
    t_statistic[i, 1], p_value[i, 1] = ttest_rel(aggregated_results[i][1], aggregated_results[i][0])

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

advantage = np.zeros((20, 2))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)

significance = np.zeros((20, 2))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table, "\n\n")
