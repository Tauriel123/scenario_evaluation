import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 绘图：聚类数随着场景库容量的变化
def n_clusters_iter(df_label):
    n = len(df_label)
    label = df_label.values
    label = label[:, 1]
    space = 10
    result, x = [], []
    for i in range(0, n, space):
        x.append(i)
        tmp = set(label[:i])
        result.append(len(tmp))
    plt.plot(x, result)
    plt.xlabel('number of scenarios')
    plt.ylabel('number of clusters')
    plt.show()


def cluster_shape(df_scenario, type, df_label):
    label = df_label.values
    label = label[:, 1]
    result = pd.DataFrame(label)
    result.columns = ['type']
    scenarios = pd.concat([df_scenario, result], axis=1)
    scenarios = scenarios.loc[scenarios['type'] == type]
    scenarios = scenarios.to_numpy()
    feature1 = scenarios[:, :50]
    feature2 = scenarios[:, 50:100]
    feature3 = scenarios[:, 100:]
    plt.subplot(1, 3, 1)
    for i in range(len(feature1)):
        plt.plot(feature1[i])
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title('feature1')
    plt.subplot(1, 3, 2)
    for i in range(len(feature2)):
        plt.plot(feature2[i])
        plt.xlabel('time')
        plt.title('feature2')
    plt.subplot(1, 3, 3)
    for i in range(len(feature3)):
        plt.plot(feature3[i])
        plt.xlabel('time')
        plt.title('feature3')
    plt.show()


# df_scenario = pd.read_csv("./data/output/freeDrive_features.csv")
# df_label = pd.read_csv("./data/output/freeDrive_label.csv")
# cluster_shape(df_scenario, -1, df_label)


import collections

df_scenario = pd.read_csv("./data/output/all_label1.csv")
df_scenario = np.array(df_scenario)
df = df_scenario[:, 1]
cnt = collections.Counter(df)
print(cnt.items())
