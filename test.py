# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # 绘图：聚类数随着场景库容量的变化
# def n_clusters_iter(df_label):
#     n = len(df_label)
#     label = df_label.values
#     label = label[:, 1]
#     space = 10
#     result, x = [], []
#     for i in range(0, n, space):
#         x.append(i)
#         tmp = set(label[:i])
#         result.append(len(tmp))
#     plt.plot(x, result)
#     plt.xlabel('number of scenarios')
#     plt.ylabel('number of clusters')
#     plt.show()
#
#
# def cluster_shape(df_scenario, type, df_label):
#     label = df_label.values
#     label = label[:, 1]
#     result = pd.DataFrame(label)
#     result.columns = ['type']
#     scenarios = pd.concat([df_scenario, result], axis=1)
#     scenarios = scenarios.loc[scenarios['type'] == type]
#     scenarios = scenarios.to_numpy()
#     feature1=scenarios[:,:50]
#     feature2 = scenarios[:, 50:100]
#     feature3 = scenarios[:, 100:]
#     plt.subplot(1, 3, 1)
#     for i in range(len(feature1)):
#         plt.plot(feature1[i])
#         plt.xlabel('time')
#         plt.ylabel('value')
#         plt.title('feature1')
#     plt.subplot(1, 3, 2)
#     for i in range(len(feature2)):
#         plt.plot(feature2[i])
#         plt.xlabel('time')
#         plt.title('feature2')
#     plt.subplot(1, 3, 3)
#     for i in range(len(feature3)):
#         plt.plot(feature3[i])
#         plt.xlabel('time')
#         plt.title('feature3')
#     plt.show()
#
#
#
# df_scenario = pd.read_csv("../Deep-temporal-clustering-main/data/NGSIM/scenarios_without.csv")
# df_label = pd.read_csv("../Deep-temporal-clustering-main/data/NGSIM/scenarios_label.csv")
# cluster_shape(df_scenario, 0, df_label)

import numpy as np

# MaxFrontDistance = 100


# all1 = pd.read_csv(
#     'C:/Users/tauriel/OneDrive/桌面/寒假研究进展/NGSIM数据/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Data/DATA (NO MOTORCYCLES).txt',
#     sep='\t')
# f = open('C:/Users/tauriel/OneDrive/桌面/寒假研究进展/NGSIM数据/scenarios_without.csv','w')
# EightModel = ['Time', 'egoID', 'LeftFront', 'Front', 'RightFront', 'Left', 'Right', 'LeftFollow', 'Follow',
#               'RightFollow']
# def feature_calculate(all1, scenario, Ego, time, index):  # index表示周围车编号
#     VehicleID = scenario[index, time]
#     Y_diff, V_diff, A_diff = 10, 10, 10
#     if int(float(VehicleID)) > 0:
#         data = all1[all1[:, 0] == int(float(VehicleID)), :]
#         data = data[data[:, 1] == time, :]
#         localY, local_V, local_Acc = data[0, 3], data[0, 4], data[0, 5]
#         Y_diff = 10 * abs(localY - Ego[0, 3]) / MaxFrontDistance
#         V_diff = abs(local_V - Ego[0, 4])
#         A_diff = abs(local_Acc - Ego[0, 5])
#     return Y_diff, V_diff, A_diff
#
#
# def feature_select(all1, scenes, paras, ScenarioDuration):
#     #  all_features待选集，可增加
#     # paras样例['speed1']
#     ego = ['speed', 'x', 'y', 'acc']
#     speed_diff = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     acc_diff = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     x_diff = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     y_diff = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     Dim = len(paras)
#     allInFo = []
#     for l in range(np.shape(scenes)[0]):
#         info = 10 * np.ones((Dim, ScenarioDuration))
#         scenario = scenes[l, :]
#         scenario = scenario.reshape(-1, ScenarioDuration)
#         EgoData = all1[all1[:, 0] == int(float(scenario[1, 0])), :]
#         T, a, b, c = np.unique(scenario[0, :], return_counts=True, return_index=True, return_inverse=True)
#         scenario_origin = scenario.astype(float)
#         TimeList = list(scenario_origin[0, :])
#         for n in T:
#             Ego = EgoData[EgoData[:, 1] == int(float(n)), :]
#             indexList = [3]
#             time = TimeList.index(n)
#             feature_id=0
#             for index in indexList:
#                 Y_diff, V_diff, A_diff = feature_calculate(all1, scenario, Ego, time, index)
#                 info[feature_id,time],info[feature_id+1,time],info[feature_id+2,time]=Y_diff, V_diff, A_diff
#                 feature_id+=3
#         info = np.array(info)
#         info = info.reshape(1, -1)
#         allInFo = allInFo + list(info)
# import pandas as pd
# a=pd.DataFrame([[1,2,3],[4,5,6]],columns=['one', 'two', 'three'])
# print(a)

# # 生成可以转换成openscenario文件的csv
# import numpy as np
# import pandas as pd
#
#
# def openS_csv(scene, all1):
#     # scene = scenes[scenario_index, :]
#     start = int(float(scene[0, 0]))
#     end = int(float(scene[0, -1]))
#     index = scene[1, 0]
#     V, a, b, c = np.unique(scene[3:, :], return_counts=True, return_index=True, return_inverse=True)
#     dataS = []
#     for vehicle in V:
#         data = all1[all1[:, 0] == int(vehicle), :]
#         data = data[data[:, 1] >= start, :]
#         data = data[data[:, 1] <= end, :]
#         data = data[:, 0:4]
#         if not dataS:
#             dataS = data
#         else:
#             dataS = np.r_[dataS, data]
#     dataS = list(dataS)
#     df = pd.DataFrame(dataS, columns=['ID', 'Time', 'PositionX', 'PositionY'])
#     df.to_csv(
#         'C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\code\\code\\Deep-temporal-clustering-main-pytorch-DBSCAN\\Deep'
#         '-temporal-clustering-main-pytorch-V2\\Deep-temporal-clustering-main\\data\\NGSIM\\' + str(
#             index) + '.xosc')

# import pandas as pd
# a=pd.DataFrame([[1,2,3],[4,5,6]],columns=['Index', 'two', 'three'])
# a.to_csv('test.csv',index=False)

# def func(a,b):
#     return a//b
#
# #算法index和路径文件
# data=[()]
# def test():
#     data=[(1,1),(1,0),(2,1)]
#     for i,j in data:
#         try:
#             res=func(i,j)
#             print(res)
#         except Exception as e:
#             print(e)  # 打印异常说明
#
# test()
import pandas as pd

# all1 = pd.read_csv(
#     'C:/Users/tauriel/OneDrive/桌面/寒假研究进展/NGSIM数据/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Data/DATA (NO MOTORCYCLES).txt',
#     sep='\t')
# all1.astype(int)
# all1 = np.array(all1)
# data = all1[all1[:, 0] == 11, :]
# print(data)
# import numpy as np
# a=np.array([1,2,3],[2,3,4])
# b=np.array([0,1,2])
# # print(b/a)
# a[(1,2),2]=[1,2]
# print(a)

all1 = pd.read_csv('./data/scenarios.csv', sep='\t')
cutIn = pd.read_csv('./data/else.txt', sep='\t')
print(len(cutIn))
