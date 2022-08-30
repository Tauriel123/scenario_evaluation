import math
import pandas as pd
import copy
import xml.etree.ElementTree as ET


# TODO：进行TREE的美化，即对根节点下的每个子节点进行相应的换行与缩进操作
def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    '''
    该部分的美化操作，置于整个tree构建完成后，进行统一美化
    :param arguments: 传进来的Elment类，缩进参数，换行参数
    :return: None

    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write('.xdor')  # 美化完成后将XML写入.xdor
    '''
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素，说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


# TODO：自动化实现OpenS转换
def xosc_write_V3(csv_path, xodr_path, output_path):
    '''
    该方法用于自动化实现场景的csv文件到openScenario格式的自动转换
    转换思路：将每辆车的轨迹按照轨迹逐帧记录storyBoard中的Event中
    【V3：针对各车时间窗口进行了统一，若非ego时间窗大于ego，则进行轨迹裁剪，否则对非EGO进行轨迹延拓；并将车辆尺寸与HighD数据进行了统一，且x坐标也修正至车辆中心】

    Input：轨迹csv文本路径、OpenDrive文件路径（一般直接是文件名.xodr，播放时与xosc放在同一个目录下）、输出的OpenScenario文件路径、主车ID
    Output：None
    '''
    root = ET.Element('OpenSCENARIO')
    # root下第一层目录构建——Level 1
    header = ET.SubElement(root, 'FileHeader')
    header.attrib = {'revMajor': '1', 'revMinor': '0', 'date': '2021-11-02T16:20:00', 'description': 'scenario_highD', 'author': 'OnSite_TOPS'}
    pareDecl = ET.SubElement(root, 'ParameterDeclarations')
    pareDecl.attrib = {}
    catalog = ET.SubElement(root, 'CatalogLocations')
    catalog.attrib = {}
    roadNet = ET.SubElement(root, 'RoadNetwork')
    roadNet.attrib = {}
    entity = ET.SubElement(root, 'Entities')
    entity.attrib = {}
    storyboard = ET.SubElement(root, 'Storyboard')
    storyboard.attrib = {}

    # root下第二层目录构建——Level 2
    logic = ET.SubElement(roadNet, 'LogicFile')
    logic.attrib = {'filepath': xodr_path}
    init = ET.SubElement(storyboard, 'Init')
    init.attrib = {}
    story = ET.SubElement(storyboard, 'Story')
    story.attrib = {'name': 'Cutin'}
    stoptrigger = ET.SubElement(storyboard, 'StopTrigger')
    stoptrigger.attrib = {}

    # root下第三层目录构建——Level 3
    actions = ET.SubElement(init, 'Actions')
    actions.attrib = {}
    paramDecl = ET.SubElement(story, 'ParameterDeclarations')
    paramDecl.attrib = {}

    # Level 4及以下，以模块为单位进行树结构构建
    '''Init-Action下的GlobalAction块'''
    globalaction = ET.Element('GlobalAction', {})
    actions.append(globalaction)
    environmentaction = ET.Element('EnvironmentAction', {})
    globalaction.append(environmentaction)
    environment = ET.Element('Environment', {'name': 'Default_Environment'})
    environmentaction.append(environment)
    timeofday = ET.Element('TimeOfDay', {'animation': 'false', 'dateTime': '2021-12-10T11:00:00'})
    environment.append(timeofday)
    weather = ET.Element('Weather', {'cloudState': 'free'})
    environment.append(weather)
    sun = ET.SubElement(weather, 'Sun')
    sun.attrib = {'intensity': '1.0', 'azimuth': '0.0', 'elevation': '1.571'}
    fog = ET.SubElement(weather, 'Fog')
    fog.attrib = {'visualRange': '100000.0'}
    precip = ET.SubElement(weather, 'Precipitation')
    precip.attrib = {'precipitationType': 'dry', 'intensity': '0.0'}
    roadcondi = ET.Element('RoadCondition', {'frictionScaleFactor': '1.0'})
    environment.append(roadcondi)
    ''''读取场景cvs后，自动化完成车辆初始化及轨迹设置'''
    sample = 1 / 10  # EKT无人机数据的轨迹采样率
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['ID'], sort=False)  # 将df按照车辆id进行group
    ego_frames = []  # 用于作为非ego轨迹延拓的baseline
    count = 0  # 用于区分主车及编号
    for group_id, df_rows in grouped:
        frame_list = df_rows['Time'].values.tolist()
        x_list = df_rows['PositionX'].values.tolist()
        y_list = df_rows['PositionY'].values.tolist()
        # v_list = df_rows['xVelocity'].values.tolist()
        if count == 0:  # 表中第一辆车为主车的情况，使用这个判断
            df_xlist = pd.DataFrame(x_list, columns=[group_id])
            df_ylist = pd.DataFrame(y_list, columns=[group_id * -1])  # 避免两个df列索引标签相同
            ego_frames = copy.deepcopy(frame_list)
            base_frame = frame_list[0]  # ego的初始帧
            whole_time = (frame_list[-1] - frame_list[0]) * sample  # 场景总时间
            # 申明Entities及对应属性
            scenObj = ET.SubElement(entity, 'ScenarioObject')
            scenObj.attrib = {'name': 'Ego'}
            veh = ET.SubElement(scenObj, 'Vehicle')
            veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car'}
            boundingbox = ET.SubElement(veh, 'BoundingBox')  # 车辆边框属性设置
            center = ET.SubElement(boundingbox, 'Center')  # 车辆中心在【车辆坐标系】中的坐标
            center.attrib = {'x': '%.16e' % 1.5, 'y': '%.16e' % 0, 'z': '%.16e' % 0.9}
            dimension = ET.SubElement(boundingbox, 'Dimensions')
            dimension.attrib = {'width': '%.16e' % 2.1, 'length': '%.16e' % 4.5, 'height': '%.16e' % 1.8}
            performance = ET.SubElement(veh, 'Performance')
            performance.attrib = {'maxSpeed': "200", 'maxAcceleration': "200", 'maxDeceleration': "10.0"}
            axles = ET.SubElement(veh, 'Axles')
            axles.attrib = {}
            front = ET.SubElement(axles, 'FrontAxle')
            front.attrib = {'maxSteering': "0.5", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "2.8", 'positionZ': "0.25"}
            rear = ET.SubElement(axles, 'RearAxle')
            rear.attrib = {'maxSteering': "0.0", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "0.0", 'positionZ': "0.25"}
            property = ET.SubElement(veh, 'Properties')
            property.attrib = {}
            # Init部分对车辆的属性设置
            private = ET.Element('Private', {'entityRef': 'Ego'})  # 申明专属对象
            actions.append(private)

            privateAction1_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作1（速度）
            privateAction1_init.attrib = {}
            longitAction = ET.SubElement(privateAction1_init, 'LongitudinalAction')
            longitAction.attrib = {}
            speedAction = ET.SubElement(longitAction, 'SpeedAction')
            speedAction.attrib = {}
            speedAcDy = ET.SubElement(speedAction, 'SpeedActionDynamics')
            speedAcDy.attrib = {'dynamicsShape': 'step', 'value': '0', 'dynamicsDimension': 'time'}
            speedAcTar = ET.SubElement(speedAction, 'SpeedActionTarget')
            speedAcTar.attrib = {}
            absTarSpeed = ET.SubElement(speedAcTar, 'AbsoluteTargetSpeed')
            v = (((x_list[1] - x_list[0])**2 + (y_list[1] - y_list[0])**2)**0.5) * sample
            absTarSpeed.attrib = {'value': '%.16e' % v}

            privateAction2_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作2（位置）
            privateAction2_init.attrib = {}
            telepAction = ET.SubElement(privateAction2_init, 'TeleportAction')
            telepAction.attrib = {}
            position_init = ET.SubElement(telepAction, 'Position')
            position_init.attrib = {}
            worldPos_init = ET.SubElement(position_init, 'WorldPosition')  # 采用全局世界坐标对车辆进行定位
            heading = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0])
            worldPos_init.attrib = {
                'x': '%.16e' % x_list[0], 'y': '%.16e' % y_list[0], 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            # Stroy部分对车辆动作（轨迹跟随）的设置
            '''
            OpenSCENARIO中通过StoryBoard展现场景的机制：
            story下设置各个车辆的动作集Act，每个Act下定义车辆对应的操作集ManeuverGroup及其触发器StartTrigger
            ManeuverGroup下定义该操作集的执行者Actor及对应的事件Event
            Event下定义具体的车辆动作Action及其触发器StartTrigger
            【对于一个Action，只有动作集Act的触发器触发并且对应Event的触发器也触发，才会执行该动作Action】
            【Act的触发要早于Event，否则仿真将出错，故下面Act的触发时间为0，Event触发时间后移一帧0.05】
            '''
            act = ET.SubElement(story, 'Act')
            act.attrib = {'name': 'Act_Ego'}
            '''车辆操作组ManeuverGroup设置'''
            maneuGroup = ET.SubElement(act, 'ManeuverGroup')
            maneuGroup.attrib = {'maximumExecutionCount': '1', 'name': 'Sequence_Ego'}
            actor = ET.SubElement(maneuGroup, 'Actors')  # 操作执行者设置
            actor.attrib = {'selectTriggeringEntities': 'false'}
            entityRef = ET.SubElement(actor, 'EntityRef')
            entityRef.attrib = {'entityRef': 'Ego'}
            maneuver = ET.SubElement(maneuGroup, 'Maneuver')  # 具体操作设置（通过事件Event的触发）
            maneuver.attrib = {'name': 'Maneuver1'}
            event = ET.SubElement(maneuver, 'Event')
            event.attrib = {'name': 'Event1', 'priority': 'overwrite'}
            action = ET.SubElement(event, 'Action')  # Event下定义的具体车辆动作
            action.attrib = {'name': 'Action1'}
            privateAction_story = ET.SubElement(action, 'PrivateAction')
            privateAction_story.attrib = {}
            routAction = ET.SubElement(privateAction_story, 'RoutingAction')
            routAction.attrib = {}
            followTraAction = ET.SubElement(routAction, 'FollowTrajectoryAction')  # 路径行为模型为轨迹跟随
            followTraAction.attrib = {}
            trajectory = ET.SubElement(followTraAction, 'Trajectory')  # 轨迹跟随的具体轨迹设置
            trajectory.attrib = {'name': 'Trajectory_Ego', 'closed': 'false'}
            shape = ET.SubElement(trajectory, 'Shape')  # 轨迹线型设置（轨迹点之间相连的方式）
            shape.attrib = {}
            polyline = ET.SubElement(shape, 'Polyline')  # 直线相连，highD轨迹为25HZ的采样
            polyline.attrib = {}
            for i in range(len(x_list) - 1):  # 批量填充csv场景中的轨迹点（去除掉初始化点）
                x = x_list[i]
                y = y_list[i]
                heading = math.atan2(y_list[i + 1] - y_list[i], x_list[i + 1] - x_list[i])
                vertex = ET.SubElement(polyline, 'Vertex')
                vertex.attrib = {'time': str(sample * (i))}
                position_story = ET.SubElement(vertex, 'Position')
                position_story.attrib = {}
                worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                worldPos_story.attrib = {
                    'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            timeRef = ET.SubElement(followTraAction, 'TimeReference')  # 轨迹跟随的时间设置
            timeRef.attrib = {}
            timing = ET.SubElement(timeRef, 'Timing')  # 选择绝对时间，（不能选择相对事件触发的时间，为保证Act先触发，Event触发点延后了0.03秒）
            timing.attrib = {'domainAbsoluteRelative': 'absolute', 'scale': '1.0', 'offset': '0.0'}
            trajecFolloeMode = ET.SubElement(followTraAction, 'TrajectoryFollowingMode')
            trajecFolloeMode.attrib = {'followingMode': 'follow'}
            startTrig_event = ET.SubElement(event, 'StartTrigger')  # Event的触发器StartTrigger
            startTrig_event.attrib = {}
            conditionGroup_event = ET.SubElement(startTrig_event, 'ConditionGroup')
            conditionGroup_event.attrib = {}
            condition_event = ET.SubElement(conditionGroup_event, 'Condition')
            condition_event.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'none'}  # 触发机制为none，即满足condi便触发（防止不同平台仿真器仿真精度不同，从而导致无法触发）
            byValueCondi_event = ET.SubElement(condition_event, 'ByValueCondition')  # 通过变量值判断条件
            byValueCondi_event.attrib = {}
            simulationTimeCondi_event = ET.SubElement(byValueCondi_event, 'SimulationTimeCondition')  # 基于仿真时间触发
            simulationTimeCondi_event.attrib = {'value': '0.03', 'rule': 'greaterThan'}
            '''车辆动作集Act的触发器设置'''
            startTrig_act = ET.SubElement(act, 'StartTrigger')  # 动作集Act触发器设置
            startTrig_act.attrib = {}
            conditionGroup_act = ET.SubElement(startTrig_act, 'ConditionGroup')
            conditionGroup_act.attrib = {}
            condition_act = ET.SubElement(conditionGroup_act, 'Condition')
            condition_act.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}
            byValueCondi_act = ET.SubElement(condition_act, 'ByValueCondition')
            byValueCondi_act.attrib = {}
            simulationTimeCondi_act = ET.SubElement(byValueCondi_act, 'SimulationTimeCondition')
            simulationTimeCondi_act.attrib = {'value': '0', 'rule': 'greaterThan'}
        else:
            # 申明Entities及对应属性
            scenObj = ET.SubElement(entity, 'ScenarioObject')
            scenObj.attrib = {'name': str('A' + str(group_id))}
            veh = ET.SubElement(scenObj, 'Vehicle')
            veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car'}
            boundingbox = ET.SubElement(veh, 'BoundingBox')  # 车辆边框属性设置
            center = ET.SubElement(boundingbox, 'Center')  # 车辆中心在【车辆坐标系】中的坐标
            center.attrib = {'x': '%.16e' % 1.5, 'y': '%.16e' % 0, 'z': '%.16e' % 0.9}
            dimension = ET.SubElement(boundingbox, 'Dimensions')
            dimension.attrib = {'width': '%.16e' % 2.1, 'length': '%.16e' % 4.5, 'height': '%.16e' % 1.8}
            performance = ET.SubElement(veh, 'Performance')
            performance.attrib = {'maxSpeed': "200", 'maxAcceleration': "200", 'maxDeceleration': "10.0"}
            axles = ET.SubElement(veh, 'Axles')
            axles.attrib = {}
            front = ET.SubElement(axles, 'FrontAxle')
            front.attrib = {'maxSteering': "0.5", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "2.8", 'positionZ': "0.25"}
            rear = ET.SubElement(axles, 'RearAxle')
            rear.attrib = {'maxSteering': "0.0", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "0.0", 'positionZ': "0.25"}
            property = ET.SubElement(veh, 'Properties')
            property.attrib = {}
            # Init部分对车辆的初始化
            '''
            考虑LG SVL仿真系统的场景需求，需要对非ego的轨迹进行延拓，使其在时间窗口上与ego保持一致
            轨迹延拓方案采用x坐标的线性外推
            '''
            if len(frame_list) != len(ego_frames):  # 时间窗口与ego不一致，需要进行轨迹修正
                # 对于时间窗口大于ego的，进行轨迹剪切
                if ego_frames[0] > frame_list[0]:
                    x_list = x_list[(ego_frames[0] - frame_list[0]):-1]
                    y_list = y_list[(ego_frames[0] - frame_list[0]):-1]
                    frame_list = frame_list[(ego_frames[0] - frame_list[0]):-1]
                if ego_frames[-1] < frame_list[-1]:
                    x_list = x_list[0:(-1 - int(frame_list[-1]) + int(ego_frames[-1]))]
                    y_list = y_list[0:(-1 - int(frame_list[-1]) + int(ego_frames[-1]))]
                    frame_list = frame_list[0:(-1 - int(frame_list[-1]) + int(ego_frames[-1]))]
                # 对于时间窗口小于ego的，进行轨迹延拓
                if x_list[0] < x_list[-1]:  # 下行场景
                    if ego_frames[0] < frame_list[0]:  # 需要前推延拓
                        temp_x = x_list[0]
                        v = (((x_list[1] - x_list[0])**2 + (y_list[1] - y_list[0])**2)**0.5) * sample
                        y = y_list[0]
                        for i in range(frame_list[0] - ego_frames[0]):
                            temp_x = temp_x - v * sample
                            x_list.insert(0, temp_x)
                            y_list.insert(0, y)
                    if ego_frames[-1] > frame_list[-1]:  # 需要后推延拓
                        temp_x = x_list[-1]
                        v = (((x_list[-1] - x_list[-2])**2 + (y_list[-1] - y_list[-2])**2)**0.5) * sample
                        y = y_list[-1]
                        for i in range(ego_frames[-1] - frame_list[-1]):
                            temp_x = temp_x + v * sample
                            x_list.append(temp_x)
                            y_list.append(y)
                else:  # 上行场景
                    if ego_frames[0] < frame_list[0]:  # 需要前推延拓
                        temp_x = x_list[0]
                        v = (((x_list[1] - x_list[0])**2 + (y_list[1] - y_list[0])**2)**0.5) * sample
                        y = y_list[0]
                        for i in range(frame_list[0] - ego_frames[0]):
                            temp_x = temp_x + v * sample
                            x_list.insert(0, temp_x)
                            y_list.insert(0, y)
                    if ego_frames[-1] > frame_list[-1]:  # 需要后推延拓
                        temp_x = x_list[-1]
                        v = (((x_list[-1] - x_list[-2])**2 + (y_list[-1] - y_list[-2])**2)**0.5) * sample
                        y = y_list[-1]
                        for i in range(ego_frames[-1] - frame_list[-1]):
                            temp_x = temp_x - v * sample
                            x_list.append(temp_x)
                            y_list.append(y)
            df_xlist = pd.concat([df_xlist, pd.DataFrame(x_list, columns=[group_id])], axis=1)
            df_ylist = pd.concat([df_ylist, pd.DataFrame(y_list, columns=[group_id * -1])], axis=1)
            private = ET.Element('Private', {'entityRef': str('A' + str(group_id))})  # 申明专属对象
            actions.append(private)

            privateAction1_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作1（速度）
            privateAction1_init.attrib = {}
            longitAction = ET.SubElement(privateAction1_init, 'LongitudinalAction')
            longitAction.attrib = {}
            speedAction = ET.SubElement(longitAction, 'SpeedAction')
            speedAction.attrib = {}
            speedAcDy = ET.SubElement(speedAction, 'SpeedActionDynamics')
            speedAcDy.attrib = {'dynamicsShape': 'step', 'value': '0', 'dynamicsDimension': 'time'}
            speedAcTar = ET.SubElement(speedAction, 'SpeedActionTarget')
            speedAcTar.attrib = {}
            absTarSpeed = ET.SubElement(speedAcTar, 'AbsoluteTargetSpeed')
            v = (((x_list[1] - x_list[0])**2 + (y_list[1] - y_list[0])**2)**0.5) * sample
            absTarSpeed.attrib = {'value': '%.16e' % v}

            privateAction2_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作2（位置）
            privateAction2_init.attrib = {}
            telepAction = ET.SubElement(privateAction2_init, 'TeleportAction')
            telepAction.attrib = {}
            position_init = ET.SubElement(telepAction, 'Position')
            position_init.attrib = {}
            worldPos_init = ET.SubElement(position_init, 'WorldPosition')  # 采用全局世界坐标对车辆进行定位
            heading = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0])
            worldPos_init.attrib = {
                'x': '%.16e' % x_list[0], 'y': '%.16e' % y_list[0], 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            # Stroy部分对车辆动作（轨迹跟随）的设置
            act = ET.SubElement(story, 'Act')
            act.attrib = {'name': str('Act_' + 'A' + str(group_id))}
            '''车辆操作组ManeuverGroup设置'''
            maneuGroup = ET.SubElement(act, 'ManeuverGroup')
            maneuGroup.attrib = {'maximumExecutionCount': '1', 'name': str('Squence_' + 'A' + str(group_id))}
            actor = ET.SubElement(maneuGroup, 'Actors')  # 操作执行者设置
            actor.attrib = {'selectTriggeringEntities': 'false'}
            entityRef = ET.SubElement(actor, 'EntityRef')
            entityRef.attrib = {'entityRef': str('A' + str(group_id))}
            maneuver = ET.SubElement(maneuGroup, 'Maneuver')  # 具体操作设置（通过事件Event的触发）
            maneuver.attrib = {'name': 'Maneuver1'}
            event = ET.SubElement(maneuver, 'Event')
            event.attrib = {'name': 'Event1', 'priority': 'overwrite'}
            action = ET.SubElement(event, 'Action')  # Event下定义的具体车辆动作
            action.attrib = {'name': 'Action1'}
            privateAction_story = ET.SubElement(action, 'PrivateAction')
            privateAction_story.attrib = {}
            routAction = ET.SubElement(privateAction_story, 'RoutingAction')
            routAction.attrib = {}
            followTraAction = ET.SubElement(routAction, 'FollowTrajectoryAction')  # 路径行为模型为轨迹跟随
            followTraAction.attrib = {}
            trajectory = ET.SubElement(followTraAction, 'Trajectory')  # 轨迹跟随的具体轨迹设置
            trajectory.attrib = {'name': str('Trajectory_' + 'A' + str(group_id)), 'closed': 'false'}
            shape = ET.SubElement(trajectory, 'Shape')  # 轨迹线型设置（轨迹点之间相连的方式）
            shape.attrib = {}
            polyline = ET.SubElement(shape, 'Polyline')  # 直线相连，highD轨迹为25HZ的采样
            polyline.attrib = {}
            '''
            考虑到非ego最多有八辆，且提取出的轨迹对应的起始帧不一定一致，需要对其轨迹进行相应修改
            初始帧晚于ego的，相差的这段时间控制其轨迹保持在原地
            '''
            if len(x_list) == len(ego_frames):  # 与ego时间窗口长度一致
                for i in range(len(x_list)):  # 批量填充csv场景中的轨迹点（去除掉初始化点）
                    x = x_list[i]
                    y = y_list[i]
                    vertex = ET.SubElement(polyline, 'Vertex')
                    vertex.attrib = {'time': str(sample * (i))}
                    position_story = ET.SubElement(vertex, 'Position')
                    position_story.attrib = {}
                    worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                    worldPos_story.attrib = {
                        'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % 0, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            else:
                # worldPos_init.set('x', '%.16e' % -10)
                stopTime = (frame_list[0] - base_frame) * sample
                vertex = ET.SubElement(polyline, 'Vertex')
                vertex.attrib = {'time': str(stopTime)}
                position_story = ET.SubElement(vertex, 'Position')
                position_story.attrib = {}
                worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                worldPos_story.attrib = {
                    'x': '%.16e' % x_list[0], 'y': '%.16e' % y_list[0], 'z': '%.16e' % 0, 'h': '%.16e' % 0, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                for i in range(len(x_list) - 1):  # 批量填充csv场景中的轨迹点（去除掉初始化点）
                    x = x_list[i + 1]
                    y = y_list[i + 1]
                    heading = math.atan2(y_list[i + 1] - y_list[i], x_list[i + 1] - x_list[i])
                    vertex = ET.SubElement(polyline, 'Vertex')
                    vertex.attrib = {'time': str(stopTime + sample * (i + 1))}
                    position_story = ET.SubElement(vertex, 'Position')
                    position_story.attrib = {}
                    worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                    worldPos_story.attrib = {
                        'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            timeRef = ET.SubElement(followTraAction, 'TimeReference')  # 轨迹跟随的时间设置
            timeRef.attrib = {}
            timing = ET.SubElement(timeRef, 'Timing')
            timing.attrib = {'domainAbsoluteRelative': 'absolute', 'scale': '1.0', 'offset': '0.0'}
            trajecFolloeMode = ET.SubElement(followTraAction, 'TrajectoryFollowingMode')
            trajecFolloeMode.attrib = {'followingMode': 'follow'}
            startTrig_event = ET.SubElement(event, 'StartTrigger')  # Event的触发器StartTrigger
            startTrig_event.attrib = {}
            conditionGroup_event = ET.SubElement(startTrig_event, 'ConditionGroup')
            conditionGroup_event.attrib = {}
            condition_event = ET.SubElement(conditionGroup_event, 'Condition')
            condition_event.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'none'}  # 触发机制为none，即满足condi便触发（防止不同平台仿真器仿真精度不同，从而导致无法触发）
            byValueCondi_event = ET.SubElement(condition_event, 'ByValueCondition')  # 通过变量值判断条件
            byValueCondi_event.attrib = {}
            simulationTimeCondi_event = ET.SubElement(byValueCondi_event, 'SimulationTimeCondition')  # 基于仿真时间触发
            simulationTimeCondi_event.attrib = {'value': '0.03', 'rule': 'greaterThan'}
            '''车辆动作集Act的触发器设置'''
            startTrig_act = ET.SubElement(act, 'StartTrigger')  # 动作集Act触发器设置
            startTrig_act.attrib = {}
            conditionGroup_act = ET.SubElement(startTrig_act, 'ConditionGroup')
            conditionGroup_act.attrib = {}
            condition_act = ET.SubElement(conditionGroup_act, 'Condition')
            condition_act.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}
            byValueCondi_act = ET.SubElement(condition_act, 'ByValueCondition')
            byValueCondi_act.attrib = {}
            simulationTimeCondi_act = ET.SubElement(byValueCondi_act, 'SimulationTimeCondition')
            simulationTimeCondi_act.attrib = {'value': '0', 'rule': 'greaterThan'}
        count += 1

    # 设置整个StoryBoard的停止器StopTrigger
    conditionGroup_stop = ET.SubElement(stoptrigger, 'ConditionGroup')
    conditionGroup_stop.attrib = {}
    condition_stop = ET.SubElement(conditionGroup_stop, 'Condition')
    condition_stop.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}  # 触发机制为rising，即condi由0至1时触发
    byValueCondi_stop = ET.SubElement(condition_stop, 'ByValueCondition')  # 通过变量值判断条件
    byValueCondi_stop.attrib = {}
    simulationTimeCondi_stop = ET.SubElement(byValueCondi_stop, 'SimulationTimeCondition')  # 基于仿真时间触发
    simulationTimeCondi_stop.attrib = {'value': str(whole_time + 0.025), 'rule': 'greaterThan'}

    tree = ET.ElementTree(root)
    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return df_xlist, df_ylist


if __name__ == "__main__":
    # 输出openS文件
    xosc_write_V3('csv_for_scenario0.csv', '', "example2.xosc")
