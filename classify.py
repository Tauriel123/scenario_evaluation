import numpy as np
import sys

sys.path.append('//'.join(sys.path[0].split('//')[:-1]))


# from score_calculate import scenario_prepare, Evaluation


def classify(scene, scenario, ScenarioDuration, index):
    """
    scenario:[0]time,[1]VehicleID,[2]EgoX,[3]EgoY,[4]EgoV,[5]EgoA,[6]FrontID,[7]FrontY,[8]FrontV
    scene：时间1,时间2...时间n|维度1-t1，维度1-t2....维度1-tn|维度2...
    EightModel = ['Time', 'egoID', 'LeftFront', 'Front', 'RightFront', 'Left', 'Right', 'LeftFollow', 'Follow',
              'RightFollow']
    """
    scene = scene.reshape(-1, ScenarioDuration)
    Lane, _, _, _ = np.unique(scenario[:, 2], return_counts=True, return_index=True, return_inverse=True)
    if len(Lane) >= 2:  # 有换道行为
        lanes = list(scenario[:, 2])
        change_time = lanes.index(Lane[1])
        if scene[3, change_time] == -1 and scene[5, change_time] == -1:
            flag = 'lane changing'
        else:
            flag = 'cut in'
        # changeLane=scenario[scenario[:,2]!=scenario[0,2],:]
        # time=int(float(changeLane[0,0]))-1
    else:
        data = scenario[scenario[:, 6] != -1]
        if len(data) / ScenarioDuration > 0.9:
            flag = 'free driving'
        elif len(data) / ScenarioDuration < 0.6:
            flag = 'following'
        else:
            flag = 'else'
    return flag
