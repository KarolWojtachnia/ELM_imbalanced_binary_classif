import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# scores = np.genfromtxt("weighted_results/resultsOf_ecoli-0-3-4-7_vs_5-6.csv", delimiter=",")
scores = pd.read_csv("weighted_results/resultsOf_ecoli-0-3-4-7_vs_5-6.csv", header=None, delimiter=',')
# scores = scores.T
# metryki i metody
metrics = ["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']
ELMs = ["ELM_100_2^-5", "ELM_100_2^-2", "ELM_100_2^0", "ELM_100_2^2",
        "ELM_100_2^5", "ELM_500_2^-5", "ELM_500_2^-2", "ELM_500_2^0",
        "ELM_500_2^2", "ELM_500_2^5", "ELM_1000_2^-5", "ELM_1000_2^-2",
        "ELM_1000_2^0", "ELM_1000_2^2", "ELM_1000_2^5"]
N = scores.shape[0]

# # kat dla kazdej z osi
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]
# 
# # spider plot
# ax = plt.subplot(111, polar=True)
# 
# # pierwsza os na gorze
# ax.set_theta_offset(pi / 2)
# ax.set_theta_direction(-1)
# 
# # po jednej osi na metryke
# plt.xticks(angles[:-1], metrics)
# 
# # os y
# ax.set_rlabel_position(0)
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#            ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
#            color="grey", size=7)
# plt.ylim(0, 1)
# # Dodajemy wlasciwe ploty dla kazdej z metod
# for elm_id, elm in enumerate(ELMs):
#     values=scores[:, elm_id].tolist()
#     values += values[:1]
#     print(values)
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=elm)
# 
# # Dodajemy legende
# plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# # Zapisujemy wykres
# plt.savefig("radar", dpi=200)

width = 0.25
x = np.arange(15)
scores.columns = metrics
plt.xticks(x + width, labels=ELMs)
plt.style.use('ggplot')
scores.plot(kind='bar')
plt.ylabel('Values of metrics')
plt.savefig("bar", dpi=200)
