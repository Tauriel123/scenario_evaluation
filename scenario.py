import pandas as pd
import numpy as np
import sys

sys.path.append('//'.join(sys.path[0].split('//')[:-1]))
from score_calculate import scenario_prepare, Evaluation
from classify import classify

"""
特征提取，针对某一个时间帧
"""


def feature_calculate(all1, scenario, Ego, time, n, index):  # index表示周围车编号
    VehicleID = scenario[index, time]
    Y_diff, V_diff, A_diff = 10, 10, 10
    if int(float(VehicleID)) > 0:
        data = all1[all1[:, 0] == int(float(VehicleID)), :]
        data = data[abs(data[:, 1] - n) < 0.1, :]
        localY, local_V, local_Acc = data[0, 3], data[0, 4], data[0, 5]
        Y_diff = 10 * abs(localY - Ego[0, 3]) / MaxFrontDistance
        V_diff = abs(local_V - Ego[0, 4])
        A_diff = abs(local_Acc - Ego[0, 5])
    return Y_diff, V_diff, A_diff


"""
对一个场景，转换成openScenario
"""


def scenario_make_up(data, start, end):
    if data[0, 1] > start:
        step = abs(data[0, 3] - data[1, 3])
        data0 = [[data[0, 0], i, data[0, 2], data[0, 3] - step * (data[0, 1] - i)] for i in
                 range(start, int(float(data[0, 1])))]
        data = np.r_[data0, data]
    # if data[-1, 1] < end:
    #     step = abs(data[0, 3] - data[1, 3])
    #     data0 = [[data[0, 0], i, data[0, 2], data[0, 3] - step * (data[0, 1] - i)] for i in range(start, data[0, 1])]
    #     data = np.r_[data0, data]
    return data


def openS_csv(scene, all1, i):
    # scene = scenes[scenario_index, :]
    scene = scene.reshape(-1, ScenarioDuration)
    start = int(float(scene[0, 0]))
    end = int(float(scene[0, -1]))
    index = scene[1, 0]
    V, a, b, c = np.unique(scene[3:, :], return_counts=True, return_index=True, return_inverse=True)
    dataS = all1[all1[:, 0] == int(float(index)), :]
    dataS = dataS[dataS[:, 1] >= start, :]
    dataS = dataS[dataS[:, 1] <= end, :]
    dataS = dataS[:, 0:4]

    for vehicle in V:
        if int(float(vehicle)) != -1:
            data = all1[all1[:, 0] == int(float(vehicle)), :]
            data = data[data[:, 1] >= start, :]
            data = data[data[:, 1] <= end, :]
            data = data[:, 0:4]
            if np.shape(data)[0] <= 1:
                continue
            elif np.shape(data)[0] < 50:
                data = scenario_make_up(data, start, end)
            dataS = np.r_[dataS, data]
    label = np.array([i for i in range(np.shape(dataS)[0])])
    dataS = np.c_[label, dataS]
    dataS = list(dataS)

    print(dataS)
    df = pd.DataFrame(dataS, columns=['Index', 'ID', 'Time', 'PositionX', 'PositionY'])
    df['ID'] = df['ID'].values.astype(int)
    df['Index'] = df['Index'].values.astype(int)
    df['Time'] = df['Time'].values.astype(int)
    df.to_csv(
        'C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\csv_for_scenario\\' + str(
            i) + '.csv', index=False)
    # print(index)


# 提取每辆车的信息：EightModel
def NGDIM_EightModel():
    EightModel = ['Time', 'egoID', 'LeftFront', 'Front', 'RightFront', 'Left', 'Right', 'LeftFollow', 'Follow',
                  'RightFollow']
    for i in vehicle_all[:100]:
        print(i)
        ego = all1[all1[:, 0] == int(i), :]
        b1, s1, t1, w1 = np.unique(ego[:, 1], return_counts=True, return_index=True, return_inverse=True)
        T = b1
        for j in T:
            VehicleT = all1[all1[:, 1] == j, :]
            # 前车
            j0 = j
            j = T.tolist().index(j)

            FrontVehicleID = ego[j, 9]
            # 后车
            FollowVehicleID = ego[j, 8]
            # 前侧
            AllFrontVehicle = VehicleT[VehicleT[:, 3] > (ego[j, 3] + 5), :]
            RightFront = AllFrontVehicle[AllFrontVehicle[:, 2] == ego[j, 2] + 1, :]
            LeftFront = AllFrontVehicle[AllFrontVehicle[:, 2] == ego[j, 2] - 1, :]

            if RightFront.size < 1:
                RightFrontVehicleID = -1
            else:
                RightFrontList = list(RightFront[:, 3])
                RightFrontIndex = RightFrontList.index(min(RightFrontList))
                RightFrontVehicleID = RightFront[RightFrontIndex, 0]

            if LeftFront.size < 1:
                LeftFrontVehicleID = -1
            else:
                LeftFrontList = list(LeftFront[:, 3])
                LeftFrontIndex = LeftFrontList.index(min(LeftFrontList))
                LeftFrontVehicleID = LeftFront[LeftFrontIndex, 0]
            # 左右
            AllSideVehicle = VehicleT[VehicleT[:, 3] > ego[j, 3] - 5, :]
            AllSideVehicle = AllSideVehicle[AllSideVehicle[:, 3] < ego[j, 3] + 5, :]
            Right = AllSideVehicle[AllSideVehicle[:, 2] == ego[j, 2] + 1, :]
            Left = AllSideVehicle[AllSideVehicle[:, 2] == ego[j, 2] - 1, :]

            if Right.size < 1:
                RightVehicleID = -1
            else:
                RightList = list(abs(list(Right[:, 3]) - ego[j, 3]))
                RightListIndex = RightList.index(min(RightList))
                RightVehicleID = Right[RightListIndex, 0]
            if Left.size < 1:
                LeftVehicleID = -1
            else:
                LeftList = list(abs(list(Left[:, 3]) - ego[j, 3]))
                LeftListIndex = LeftList.index(min(LeftList))
                LeftVehicleID = Left[LeftListIndex, 0]

            # 后侧
            AllFollowVehicle = VehicleT[VehicleT[:, 3] < ego[j, 3] - 5, :]
            AllFollowVehicle = VehicleT[VehicleT[:, 3] > ego[j, 3] - 55, :]
            RightFollow = AllFollowVehicle[AllFollowVehicle[:, 2] == ego[j, 2] + 1, :]
            LeftFollow = AllFollowVehicle[AllFollowVehicle[:, 2] == ego[j, 2] - 1, :]

            if RightFollow.size < 1:
                RightFollowVehicleID = -1
            else:
                L = list(RightFollow[:, 3])
                RightFollowIndex = L.index(max(L))
                RightFollowVehicleID = RightFollow[RightFollowIndex, 0]

            if LeftFollow.size < 1:
                LeftFollowVehicleID = -1
            else:
                L = list(LeftFollow[:, 3])
                LeftFollowIndex = L.index(max(L))
                LeftFollowVehicleID = LeftFollow[LeftFollowIndex, 0]

            EightModel = EightModel + [
                j0, i, LeftFrontVehicleID, FrontVehicleID, RightFrontVehicleID, LeftVehicleID, RightVehicleID,
                LeftFollowVehicleID, FollowVehicleID, RightFollowVehicleID]
    return EightModel


def save_scenario(filename, scenarios):
    f = open(filename, 'w')
    for i in range(np.shape(scenarios)[0]):
        OneScenario = ''
        for j in range(np.shape(scenarios)[1]):
            OneScenario = OneScenario + str(scenarios[i, j])
            OneScenario = OneScenario + ' '
        OneScenario = OneScenario + '\n'
        # print(OneScenario)
        f.write(OneScenario)
    f.close()


# 读取数据，以NGSIM为例
all1 = pd.read_csv(
    'C:/Users/tauriel/OneDrive/桌面/寒假研究进展/NGSIM数据/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Reconstructed NGSIM I80-1 data/Data/DATA (NO MOTORCYCLES).txt',
    sep='\t')

all1 = np.array(all1)
time_stamp = 1 / 10

b, s, t, w = np.unique(all1[:, 0], return_counts=True, return_index=True, return_inverse=True)
# b1, s1, t1, w1 = np.unique(all1[:, 1], return_counts=True, return_index=True, return_inverse=True)
vehicle_all = b

# Parameters
MaxFrontDistance = 100
MaxFollowDistance = 50
MaxSideDistance = 5
EightModelParas = [MaxFrontDistance, MaxFollowDistance, MaxSideDistance]

# 提取八车模型
EightModel = NGDIM_EightModel()
EightModel = np.array(EightModel)
EightModel = EightModel.reshape(-1, 10)
print(np.shape(EightModel))

# 提取场景,拆分,结构为一个场景为一行，eg.场景1：时间1,时间2...时间n|维度1-t1，维度1-t2....维度1-tn|维度2...

ScenarioDuration = 50  # 0.1seconds
dt = 10
t = 0
dim = 10
EightModel = EightModel[1:, :]
totalT = np.shape(EightModel)[0]

scenes = []
while t + ScenarioDuration < totalT:
    print(t)
    scene = EightModel[t:t + ScenarioDuration, :]
    scene = scene.transpose()
    scene = scene.reshape(-1, dim * ScenarioDuration)
    b, b1, b2, b3 = np.unique(scene[0, 50:100], return_counts=True, return_index=True, return_inverse=True)
    t += dt
    if len(b) > 1:
        continue
    scenes = scenes + list(scene)

scenes = np.array(scenes)
print(np.shape(scenes))

# 提取特征序列
scenarios = []
feature_flag = 0
# 相对距离,dim=3,前方三车
if feature_flag == 1:
    indexList = [2, 3, 4]  # 要提取的特征
    # [2]'LeftFront', [3]'Front', [4]'RightFront', [5]'Left', [6]'Right', [7]'LeftFollow', [8]'Follow',
    # [9]'RightFollow'
    Dim = len(indexList) * 3
    allInFo = []
    for l in range(np.shape(scenes)[0]):
        info = 10 * np.ones((Dim, ScenarioDuration))  # 特征矩阵
        scenario = scenes[l, :]
        scenario = scenario.reshape(-1, ScenarioDuration)
        EgoData = all1[all1[:, 0] == int(float(scenario[1, 0])), :]
        T, a, b, c = np.unique(scenario[0, :], return_counts=True, return_index=True, return_inverse=True)
        scenario_origin = scenario.astype(float)
        TimeList = list(scenario_origin[0, :])

        for n in T:
            print(n)
            # if abs(int(float(n)) - 481) < 0.1:
            #     print('hello')
            n = int(float(n))
            Ego = EgoData[EgoData[:, 1] == n, :]
            if not len(Ego):
                continue
            time = TimeList.index(int(float(n)))
            feature_id = 0
            for index in indexList:
                Y_diff, V_diff, A_diff = feature_calculate(all1, scenario, Ego, time, n, index)
                info[feature_id, time], info[feature_id + 1, time], info[feature_id + 2, time] = Y_diff, V_diff, A_diff
                feature_id += 3
        info = np.array(info)
        info = info.reshape(1, -1)
        allInFo = allInFo + list(info)
    print(np.shape(allInFo))

    dim = np.shape(allInFo)[1] / int(ScenarioDuration)
    length = ScenarioDuration
    scenarios = np.array(allInFo)  # 特征数据的文件

    # 场景格式标准化，按照tslearn标准，参考：https://blog.csdn.net/qq_40206371/article/details/122686570
    # for i in range(np.shape(scenarios)[0]):
    #     OneScenario = ''
    #     for j in range(np.shape(scenarios)[1]):
    #         OneScenario = OneScenario + str(scenarios[i, j])
    #         if j % length == ScenarioDuration - 1 and j != (dim * ScenarioDuration - 1):
    #             OneScenario = OneScenario + '|'
    #         else:
    #             OneScenario = OneScenario + ' '
    #     OneScenario = OneScenario + '\n'
    #     print(OneScenario)
    #     f.write(OneScenario)
    # f.close()
    filename = './data/scenarios_all.txt'
    save_scenario(filename, scenarios)
    result = pd.DataFrame(list(scenarios))
    result.to_csv('./data/scenarios.csv')

scenario_flag = 0  # 转换openS的csv
if scenario_flag:
    for i in range(np.shape(scenes)[0]):
        # print(i)
        scene = scenes[i]
        openS_csv(scene, all1, i)
    print('end')

score_flag = 0  # 计算得分
if score_flag == 1:
    scenarios = pd.read_csv('./data/scenarios.csv')
    scenarios = np.array(scenarios)
    score = []
    for i in range(np.shape(scenes)[0]):
        scene = scenes[i]
        tmp = scenario_prepare(scene, all1, i, ScenarioDuration)
        tmp0 = Evaluation(tmp)
        print(tmp0)
        score += tmp0

    score = np.array(score)
    score = score.reshape(np.shape(scenes)[0], -1)
    result = pd.DataFrame(list(score), columns=['safe', 'efficiency'])
    result.to_csv('./data/output/score.csv')

# if feature_flag == 0:
#     scenarios = pd.read_csv('C:/Users/tauriel/OneDrive/桌面/寒假研究进展/NGSIM数据/scenarios_without.csv',sep='\t')
#     scenarios = np.array(scenarios)

classify_flag = 1  # 场景粗分类
if classify_flag == 1:
    classes = []
    for i in range(np.shape(scenes)[0]):
        scene = scenes[i]
        scenario = scenario_prepare(scene, all1, i, ScenarioDuration)  # 场景文件,包含本车和周围车，可以用于分类和打分
        flag = classify(scene, scenario, ScenarioDuration, i)
        classes.append(flag)
    result = pd.DataFrame(list(classes), columns=['classes'])
    result.to_csv('./data/output/classes.csv')
    cutIn, laneChange, freeDrive, follow, other = [], [], [], [], []
    for i in range(len(classes)):
        if classes[i] == 'cut in':
            cutIn += list(scenarios[i, :])
        elif classes[i] == 'lane changing':
            laneChange += list(scenarios[i, :])
        elif classes[i] == 'free driving':
            freeDrive += list(scenarios[i, :])
        elif classes[i] == 'following':
            follow += list(scenarios[i, :])
        else:
            other += list(scenarios[i, :])
    # 保存各类场景
    filename1 = './data/cutIn.txt'
    cutIn = np.array(cutIn)
    cutIn = cutIn.reshape(-1, np.shape(scenarios)[1])
    save_scenario(filename1, cutIn)
    filename2 = './data/laneChange.txt'
    laneChange = np.array(laneChange)
    laneChange = laneChange.reshape(-1, np.shape(scenarios)[1])
    save_scenario(filename2, laneChange)
    filename3 = './data/freeDrive.txt'
    freeDrive = np.array(freeDrive)
    freeDrive = freeDrive.reshape(-1, np.shape(scenarios)[1])
    save_scenario(filename3, freeDrive)
    filename4 = './data/follow.txt'
    follow = np.array(follow)
    follow = follow.reshape(-1, np.shape(scenarios)[1])
    save_scenario(filename4, follow)
    filename5 = './data/else.txt'
    other = np.array(other)
    other = other.reshape(-1, np.shape(scenarios)[1])
    save_scenario(filename5, other)
