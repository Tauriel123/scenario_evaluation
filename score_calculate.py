"""
Author: 陈秋冰
DateTime:2022.3.14 15:58
Version:1.0.1
"""
import numpy as np


def scenario_prepare(scene, all1, id, ScenarioDuration):
    scene = scene.reshape(-1, ScenarioDuration)
    scenario = []
    for time in range(ScenarioDuration):
        egoData = all1[all1[:, 0] == int(float(scene[1, time])), :]
        egoData = egoData[egoData[:, 1] == int(float(scene[0, time])), :]
        if int(float(scene[3, time])) == -1:
            scenario += [int(float(scene[1, time])), int(float(scene[0, time])), egoData[0, 2], egoData[0, 3],
                         egoData[0, 4],
                         egoData[0, 5], -1, -1, -1, -1]
        else:
            FrontData = all1[all1[:, 0] == int(float(scene[3, time])), :]
            FrontData = FrontData[FrontData[:, 1] == int(float(scene[0, time])), :]
            scenario += [int(float(scene[1, time])), int(float(scene[0, time])), egoData[0, 2], egoData[0, 3],
                         egoData[0, 4],
                         egoData[0, 5], FrontData[0, 0], FrontData[0, 2], FrontData[0, 3], FrontData[0, 4]]
    scenario = np.array(scenario)
    scenario = scenario.reshape(ScenarioDuration, -1)
    return scenario


def Evaluation(scenario):
    """
    scenario:[0]time,[1]VehicleID,[2]EgoX,[3]EgoY,[4]EgoV,[5]EgoA,[6]FrontID,[7]FrontY,[8]FrontV
    """
    result = []
    # 参数设置
    vehicleLength = 0  # 单位：米
    step = 0.1  # 单位:秒
    time_duration = len(scenario)
    minTTC = 2.7  # 单位:秒
    safe = 100  # 满分

    # 安全指标
    data = scenario
    crash_check = abs(data[:, 7] - data[:, 3])
    crash_flag = crash_check[crash_check < vehicleLength]
    if len(crash_flag) > 0:
        safe_score = 0
    else:
        data = data[data[:, 6] > -1, :]
        if len(data) == 0:
            safe_score = safe
        else:
            TTC = (data[:, 7] - data[:, 3]) / (data[:, 8] - data[:, 4])
            safe_score = len(TTC[TTC < minTTC]) / time_duration * safe
    result.append(safe_score)

    # 效率
    data = scenario
    efficiency_socre = data[-1, 3] - data[0, 3]
    result.append(efficiency_socre)

    return result
