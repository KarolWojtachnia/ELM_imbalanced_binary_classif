import glob
import os

import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate
names_datasets = np.array([["ecoli-0-1-3-7_vs_2-6"], ["ecoli-0-1-4-6_vs_5"], ["ecoli-0-1_vs_2-3-5"], ["ecoli-0-2-3-4_vs_5"], ["ecoli-0-3-4-7_vs_5-6"],
                        ["ecoli-0-6-7_vs_3-5"], ["ecoli-0-6-7_vs_5"], ["ecoli4"], ["glass-0-1-6_vs_2"], ["glass-0-1-6_vs_5"],
                        ["glass-0-4_vs_5"], ["glass4"], ["glass5"], ["page-blocks-1-3_vs_4"], ["shuttle-c0-vs-c4"],
                        ["shuttle-c2-vs-c4"], ["yeast-0-2-5-6_vs_3-7-8-9"], ["yeast-1-4-5-8_vs_7"], ["yeast-1_vs_7"], ["yeast-2_vs_4"]])

headers = ["no W", "W", "MLP", "SVC", "GNB"]

names_column = np.array([["no W"], ["W"], ["MLP"], ["SVC"], ["GNB"]])

results = np.load('other_Classifiers_results/all_results.npy')
mean_results = np.mean(results, axis = 2)






for i in range(20):
    # print(f"dataset name:{name.replace('resultsOf_', '').replace('.npy', '')}")
    print('\n')
    print(names_datasets[i])
    part_results = results[i, :, :]
    part_results = part_results.T
    mean_results = np.mean(part_results, axis=1)
    ranks = []
    for ms in part_results:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)

    alfa = .05
    w_statistic = np.zeros((len(headers), len(headers)))
    p_value = np.zeros((len(headers), len(headers)))

    for i in range(len(headers)):
        for j in range(len(headers)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    print(np.mean(ranks, axis=0))
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(headers), len(headers)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(headers), len(headers)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table, "\n\n")
