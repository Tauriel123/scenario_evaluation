import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import math


def write_answerSheet(root_path, answer_path, output_path):
    '''
    将参赛者输出的轨迹数据替换掉主车轨迹，并更新其前车信息

    input：标准场景根目录、参赛者轨迹、输出的答题卡路径
    output：标准答题卡
    '''
    highD_meta = os.path.abspath("./recording.csv")
    df_recording = pd.read_csv(highD_meta)  # 各类场景信息
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if '_test.csv' in file:  # 获取标准场景
                csv_path = os.path.join(root, file)
            if '.xodr' in file:  # 获取场景对应的HighD索引
                highD_index = int(file[6:-5])
    if int(highD_index) < 10:
        tracksMeta_path = os.path.join("./highD-dataset-v1.0/data/0" + str(highD_index) + "_tracksMeta.csv")
    else:
        tracksMeta_path = os.path.join("./highD-dataset-v1.0/data/" + str(highD_index) + "_tracksMeta.csv")
    df_tracksMeta = pd.read_csv(tracksMeta_path)
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['id'], sort=False)  # 将df按照车辆id进行group
    # 判断答卷是否为空
    if os.path.getsize(answer_path):
        df_answer = pd.read_csv(answer_path)
    else:
        return
    df_scenario = pd.DataFrame()
    count = 0
    for group_id, df_rows in grouped:
        if count == 0:  # ego，替换为参赛者轨迹
            ego_id = group_id
            height = df_tracksMeta[df_tracksMeta['id'] == ego_id]['height'].values.tolist()[0]  # 车宽
            ego_frame = df_rows['frame'].values.tolist()
            df_base = df_rows[['frame', 'id']]
            df_scenario = df_rows.drop(df.columns[[2, 3, 10, 11, 12, 13, 14, 15, 16, 17]], axis=1)
            # 判断主车车道
            ego_y = df_answer['y'].values.tolist()
            laneid = []
            for y in ego_y:
                upper_marking = str(df_recording[df_recording['id'] == int(highD_index)]['upperLaneMarkings'][
                                        int(highD_index) - 1]).split(';')
                upper_marking = [float(i) for i in upper_marking]
                lower_marking = str(df_recording[df_recording['id'] == int(highD_index)]['lowerLaneMarkings'][
                                        int(highD_index) - 1]).split(';')
                lower_marking = [float(i) for i in lower_marking]
                y_bias = upper_marking[-1] + (lower_marking[0] - upper_marking[-1]) / 2
                y_highD = y_bias - y - height / 2  # 将车辆纵坐标还原至HighD坐标系
                temp = upper_marking + lower_marking
                temp.append(y_highD)
                temp.sort()
                if len(upper_marking) == 3:  # 双车道场景
                    if temp.index(y_highD) == 1:
                        laneid.append(2)
                    elif temp.index(y_highD) == 2:
                        laneid.append(3)
                    elif temp.index(y_highD) == 4:
                        laneid.append(5)
                    elif temp.index(y_highD) == 5:
                        laneid.append(6)
                    else:  # 驶出行车道
                        laneid.append(-999)
                else:  # 三车道场景
                    if temp.index(y_highD) == 1:
                        laneid.append(2)
                    elif temp.index(y_highD) == 2:
                        laneid.append(3)
                    elif temp.index(y_highD) == 3:
                        laneid.append(4)
                    elif temp.index(y_highD) == 5:
                        laneid.append(6)
                    elif temp.index(y_highD) == 6:
                        laneid.append(7)
                    elif temp.index(y_highD) == 7:
                        laneid.append(8)
                    else:
                        laneid.append(-999)
            col_name = df_scenario.columns.tolist()
            col_name.insert(2, 'x')
            col_name.insert(3, 'y')
            col_name.insert(10, 'laneId')
            df_scenario = df_scenario.reindex(columns=col_name)
            df_scenario['x'] = df_answer[['x']]
            df_scenario['y'] = df_answer[['y']]
            df_scenario['laneId'] = pd.DataFrame(laneid)

            count += 1
        else:
            df_nonego = df_base.copy(deep=True)  # 深拷贝，使得复制的df拥有自己的数据与索引，而非与df_base共用
            df_nonego['id'].replace(ego_id, group_id, inplace=True)
            x = df_rows['x'].values.tolist()
            y = df_rows['y'].values.tolist()
            width = df_rows['width'].values.tolist()
            height = df_rows['height'].values.tolist()
            xv = df_rows['xVelocity'].values.tolist()
            yv = df_rows['yVelocity'].values.tolist()
            xa = df_rows['xAcceleration'].values.tolist()
            ya = df_rows['yAcceleration'].values.tolist()
            laneid = df_rows['laneId'].values.tolist()
            # 将各列表加入dataframe中
            col_name = df_scenario.columns.tolist()
            df_nonego = df_nonego.reindex(columns=col_name)
            df_nonego['x'] = x
            df_nonego['y'] = y
            df_nonego['width'] = width
            df_nonego['height'] = height
            df_nonego['xVelocity'] = xv
            df_nonego['yVelocity'] = yv
            df_nonego['xAcceleration'] = xa
            df_nonego['yAcceleration'] = ya
            df_nonego['laneId'] = laneid
            # 加入场景df
            df_scenario = pd.concat([df_scenario, df_nonego])
            count += 1
    '''
    在df中补充ego前车信息
    '''
    grouped = df_scenario.groupby(['id'], sort=False)  # 将df按照车辆id进行group
    preced = []
    preced_xv, preced_x, preced_y, preced_yv, preced_xa, preced_ya = [[] for _ in range(6)]
    ego_x = []
    ego_laneid = []
    ego_frame = []
    nonego_frame, nonego_id, nonego_x, nonego_y, nonego_laneid, nonego_xv, nonego_yv, nonego_xa, nonego_ya = [[] for _
                                                                                                              in
                                                                                                              range(9)]
    count = 0
    for group_id, rows in grouped:
        if count == 0:  # ego
            ego_frame = rows['frame'].values.tolist()
            ego_x = rows['x'].values.tolist()
            ego_laneid = rows['laneId'].values.tolist()
            count += 1
        else:
            nonego_frame.extend(rows['frame'].values.tolist())
            nonego_id.extend(rows['id'].values.tolist())
            nonego_x.extend(rows['x'].values.tolist())
            nonego_y.extend(rows['y'].values.tolist())
            nonego_laneid.extend(rows['laneId'].values.tolist())
            nonego_xv.extend(rows['xVelocity'].values.tolist())
            nonego_yv.extend(rows['yVelocity'].values.tolist())
            nonego_xa.extend(rows['xAcceleration'].values.tolist())
            nonego_ya.extend(rows['yAcceleration'].values.tolist())
            count += 1
    if ego_x[0] < ego_x[1]:  # 下行方向
        for i_ego, x_ego in enumerate(ego_x):
            temp_index = []
            temp_dis = []
            for i, x in enumerate(nonego_x):
                if nonego_frame[i] == ego_frame[i_ego] and nonego_laneid[i] == ego_laneid[
                    i_ego] and x > x_ego:  # 位于ego前方
                    temp_index.append(i)
                    temp_dis.append(x - x_ego)
            if len(temp_dis) == 0:  # ego前方无车
                preced.append(-999)
                preced_x.append(-999)
                preced_y.append(-999)
                preced_xv.append(-999)
                preced_yv.append(-999)
                preced_xa.append(-999)
                preced_ya.append(-999)
            else:
                preced_index = temp_index[temp_dis.index(min(temp_dis))]
                preced.append(nonego_id[preced_index])
                preced_x.append(nonego_x[preced_index])
                preced_y.append(nonego_y[preced_index])
                preced_xv.append(nonego_xv[preced_index])
                preced_yv.append(nonego_yv[preced_index])
                preced_xa.append(nonego_xa[preced_index])
                preced_ya.append(nonego_ya[preced_index])
    else:
        for i_ego, x_ego in enumerate(ego_x):
            temp_index = []
            temp_dis = []
            for i, x in enumerate(nonego_x):
                if nonego_frame[i] == ego_frame[i_ego] and nonego_laneid[i] == ego_laneid[
                    i_ego] and x < x_ego:  # 位于ego前方
                    temp_index.append(i)
                    # temp_dis.append(x - x_ego)
                    temp_dis.append(x_ego - x)
            if len(temp_dis) == 0:  # ego前方无车
                preced.append(-999)
                preced_x.append(-999)
                preced_y.append(-999)
                preced_xv.append(-999)
                preced_yv.append(-999)
                preced_xa.append(-999)
                preced_ya.append(-999)
            else:
                preced_index = temp_index[temp_dis.index(min(temp_dis))]
                preced.append(nonego_id[preced_index])
                preced_x.append(nonego_x[preced_index])
                preced_y.append(nonego_y[preced_index])
                preced_xv.append(nonego_xv[preced_index])
                preced_yv.append(nonego_yv[preced_index])
                preced_xa.append(nonego_xa[preced_index])
                preced_ya.append(nonego_ya[preced_index])
    # 非主车部分用-999补全
    length = len(ego_x) + len(nonego_x)
    preced = list(preced + [-999] * (length - len(preced)))
    preced_x = list(preced_x + [-999] * (length - len(preced_x)))
    preced_y = list(preced_y + [-999] * (length - len(preced_y)))
    preced_xv = list(preced_xv + [-999] * (length - len(preced_xv)))
    preced_yv = list(preced_yv + [-999] * (length - len(preced_yv)))
    preced_xa = list(preced_xa + [-999] * (length - len(preced_xa)))
    preced_ya = list(preced_ya + [-999] * (length - len(preced_ya)))
    # 将ego前车信息添加至df
    col_name = df_scenario.columns.tolist()
    col_name.append('precedingId')
    col_name.append('precedingX')
    col_name.append('precedingY')
    col_name.append('precedingXVelocity')
    col_name.append('precedingYVelocity')
    col_name.append('precedingXAcceleration')
    col_name.append('precedingYAcceleration')
    df_scenario = df_scenario.reindex(columns=col_name)
    df_scenario['precedingId'] = preced
    df_scenario['precedingX'] = preced_x
    df_scenario['precedingY'] = preced_y
    df_scenario['precedingXVelocity'] = preced_xv
    df_scenario['precedingYVelocity'] = preced_yv
    df_scenario['precedingXAcceleration'] = preced_xa
    df_scenario['precedingYAcceleration'] = preced_ya
    df_scenario = df_scenario[(df_scenario['id'] == ego_id)]

    df_scenario = df_scenario[0:-2]
    df_scenario.to_csv(output_path, index=None)

    return


def Evaluation(list_HAV, list_origin, minTTC=2.7, area=20, safety=50, efficiency=30, comfort=20,
               dt=0.04,

               Pcomfort=0.1):
    # 调整参数预设值

    if np.shape(list_origin)[0] == 0:
        print("缺乏原始场景轨迹！")
    else:
        # 安全指标
        array_HAV = list_origin
        array_HAV0 = array_HAV[array_HAV[:, 12] > -999, :]
        if np.shape(array_HAV0)[0] == 0:
            minTTC = -999
        else:
            TTC = (array_HAV0[:, 13] - array_HAV0[:, 2]) / (array_HAV0[:, 6] - array_HAV0[:, 12])
            TTC = TTC[TTC > 0]
            minTTC = min(TTC)

        # comfortable
        maxAcc = max(list_origin[:, 8])
        maxDec = min(list_origin[:, 8])
        maxAcc_lateral = max(list_origin[:, 9])
        maxDec_lateral = min(list_origin[:, 9])

        # 设置终点
        start = list_origin[0, 2]
        destination = list_origin[-1, 2]  # x坐标序列
        if start < destination:
            dest = destination - area  # 设置终点范围
            List_origin = list_origin[list_origin[:, 2] < dest, :]
        else:
            dest = destination + area
            List_origin = list_origin[list_origin[:, 2] > dest, :]
        # print("终点%d",dest)
        # 效率指标

        time_cost = 0.5*np.shape(List_origin)[0] * dt  # time_id

        # # fuel consumption，把所有油耗累计起来
        # Kij = [[-7.537, 0.4438, 0.1716, -0.0420], [0.0973, 0.0518, 0.0029, -0.0071],
        #        [-0.0030, -0.000742, 0.000109, 0.000116], [0.000053, 0.000006, -0.00001, -0.000006]]
        # fuel = 0
        # for n in range(np.shape(List_origin)[0]):
        #     for i in range(4):
        #         for j in range(4):
        #             Fuel_ij = Kij[i][j] * (abs(List_origin[n, 6]) ** i) * (List_origin[n, 8] ** j)
        #             fuel = fuel + math.exp(Fuel_ij) * dt
        # print("期望油耗%d", fuel)

    benchmark = 'ego'
    if benchmark == 'ego':
        # 本车为参考系
        # 计算score
        if start > dest:
            List_HAV = list_HAV[list_HAV[:, 2] > dest, :]
        else:
            List_HAV = list_HAV[list_HAV[:, 2] < dest, :]

        # 安全得分
        if min(abs(list_HAV[:, 2] - list_HAV[:, 12])) < 5:  # 针对阿波罗，无reward
            # if (1 in reward) or (True in reward):
            score_safe = 0
        else:
            list_HAV0 = list_HAV[list_HAV[:, 12] > -999, :]
            if np.shape(list_HAV0)[0] == 0:
                score_safe = 50
            else:
                if min(abs(list_HAV0[:, 12] - list_HAV0[:, 2])) > 5:
                    TTC_HAV = (list_HAV0[:, 13] - list_HAV0[:, 2]) / (list_HAV0[:, 6] - list_HAV0[:, 12])
                    TTC_HAV = TTC_HAV[TTC_HAV < minTTC]
                    score_safe = safety - np.shape(TTC_HAV)[0] / np.shape(List_HAV)[0] * safety
                    score_safe = min(50, score_safe)
                    score_safe = max(0, score_safe)
                else:
                    score_safe = 0

        # 效率得分
        HAV = list_HAV[list_HAV[:, 2] > dest, :]
        if not np.shape(HAV)[0]:
            score_efficiency = 0
        else:
            time_cost_HAV = np.shape(HAV)[0] * dt
            score_efficiency = min(efficiency, time_cost / time_cost_HAV * efficiency)

        # comfortable
        UncomfortAcc = List_HAV[List_HAV[:, 8] > maxAcc, 8]
        UncomfortDec = List_HAV[List_HAV[:, 8] < maxDec, 8]
        UncomfortAcc_lateral = List_HAV[List_HAV[:, 9] > maxAcc, 9]
        UncomfortDec_lateral = List_HAV[List_HAV[:, 9] < maxDec, 9]
        score_comfort = comfort - sum(UncomfortAcc - maxAcc) * Pcomfort - sum(maxDec - UncomfortDec) * Pcomfort - sum(
            UncomfortAcc_lateral - maxAcc_lateral) * Pcomfort - sum(maxDec_lateral - UncomfortDec_lateral) * Pcomfort
        score_comfort = max(0, score_comfort)
        score_comfort = min(comfort, score_comfort)


    else:  # benchmark为标准参数
        if start > dest:
            List_HAV = list_HAV[list_HAV[:, 2] > dest, :]
        else:
            List_HAV = list_HAV[list_HAV[:, 2] < dest, :]

        # 安全得分
        # if max(list_HAV[:, 14]) < 1:
        if min(abs(List_HAV[:, 12] - List_HAV[:, 2])) > 5:
            TTC_HAV = abs(List_HAV[:, 12] - List_HAV[:, 2]) / (List_HAV[:, 6] - List_HAV[:, 14])
            TTC_HAV = TTC_HAV[abs(TTC_HAV) < minTTC]
            score_safe = safety - np.shape(TTC_HAV)[0] / np.shape(List_HAV)[0] * safety
        else:
            score_safe = 0
            print('碰撞')

        # 效率得分
        HAV = list_HAV[list_HAV[:, 2] > dest, :]
        if not np.shape(HAV)[0]:
            score_efficiency = 0
        else:
            time_cost_HAV = np.shape(List_HAV)[0] * dt
            score_efficiency = min(efficiency, 0.8 * time_cost / time_cost_HAV * efficiency)

        # comfortable
        maxAcc = 1
        maxDec = -1
        UncomfortAcc = List_HAV[List_HAV[:, 8] > maxAcc, 8]
        UncomfortDec = List_HAV[List_HAV[:, 8] < maxDec, 8]
        score_comfort = comfort - sum(UncomfortAcc - maxAcc) * Pcomfort - sum(maxDec - UncomfortDec) * Pcomfort
        score_comfort = min(20, score_comfort)
        score_comfort = max(0, score_comfort)

    score = score_safe + score_efficiency + score_comfort
    print('safe'+str(score_safe))
    print('efficiency' + str(score_efficiency))
    print('comfort' + str(score_comfort))
    return score


def DataSource(scenario_name):
    filename = 'E:\测试网站\测试目标\scenario_for_OnSite\scenario_for_OnSite\index.csv'
    dataframe1 = pd.read_csv(filename, header=None)
    dataframe1 = np.array(dataframe1)
    scenario_name = scenario_name[: -5]

    source = dataframe1[dataframe1[:, 0] == scenario_name, :]
    source = source[0][2]
    return source


def write_answerSheet_NDS(root_path, result_path):
    return


def Standardization_NDS(Origin_file, scenario):
    dt = 0.1
    path = os.join(Origin_file, scenario + '_test.csv')
    answer_path = os.join(Origin_file, 'Standard.csv')
    dataframe_origin = pd.read_csv(path, header=None)
    data = np.array(dataframe_origin)
    data1 = dataframe_origin[dataframe_origin[:, 0] == 1, :]
    direction = data1[1, 2] - data1[0, 2]
    index = [x for x in range(math.floor(10 * data[-1, 1]))]
    for frame in index[1:]:
        output = []
        data_frame = data[data[:, 1] == 0.1 * frame, :]
        output.append(frame)
        output.append(data1[0, 0])
        output.append(data_frame[0, 2])
        output.append(data_frame[0, 3])
        output.append(4)
        output.append(6)  # height
        if direction > 0:
            front_vehicle = data_frame[data_frame[:, 2] > data_frame[0, 2], :]
            front_vehicle = front_vehicle[np.argsort(front_vehicle[:, 2])]
            front_vehicle = front_vehicle[0, :]
        else:
            front_vehicle = data_frame[data_frame[:, 2] < data_frame[0, 2], :]
            front_vehicle = front_vehicle[np.argsort(front_vehicle[:, 2])]
            front_vehicle = front_vehicle[-1, :]
        if frame == 0:
            output = output + [0, 0, 0, 0, 0]
            output.append(front_vehicle[0, 2])
            output.append(front_vehicle[0, 3])
            output = output + [0, 0, 0, 0]
            scenario_last = output
            scenario = np.array(output)
        else:
            output.append((output[0, 2] - scenario_last[0, 2]) / dt)
            output.append((output[0, 3] - scenario_last[0, 3]) / dt)
            output.append((output[0, 6] - scenario_last[0, 6]) / dt)
            output.append((output[0, 7] - scenario_last[0, 7]) / dt)
            output = output + [0]
            output.append(front_vehicle[0, 2])
            output.append(front_vehicle[0, 3])
            output.append(front_vehicle[0, 6])
            output = output + [0, 0, 0]
        scenario = np.r_[scenario, np.array(output)]
    np.savetxt(answer_path, scenario)
    return


if __name__ == '__main__':
    # 读取测试
    dataframe1 = pd.read_csv('./data/ans_out_origin.csv', header=None)  # 原始
    dataframe2 = pd.read_csv('./data/ans_out_auto.csv', header=None)  # 答卷
    list_HAV = dataframe2.values
    list_origin = dataframe1.values
    list_HAV = np.array(list_HAV[1:])
    list_origin = np.array(list_origin[1:])
    list_HAV = np.array(list_HAV, dtype=float)
    list_origin = np.array(list_origin, dtype=float)
    id = list_origin[0, 1]
    list_origin = list_origin[list_origin[:, 1] == id, :]
    score = Evaluation(list_HAV, list_origin, minTTC=2.7, area=0, safety=50, efficiency=30, comfort=20, dt=0.04,
                       Pcomfort=0.1)
    print(score)

