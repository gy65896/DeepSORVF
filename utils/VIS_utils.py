import csv

import numpy as np
import cv2
import torch
from PIL import Image
import pandas as pd
from detection_yolox.yolo import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from warnings import simplefilter
import cv2
from PIL import Image
import pandas as pd
from IPython import embed
import os
simplefilter(action='ignore', category=FutureWarning)
# 初始化目标检测
yolo = YOLO()

# 初始化跟踪模型
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)



def box_whether_in_area(bounding_box, Area):
    x_center = (bounding_box[0] + bounding_box[2]) / 2
    y_center = (bounding_box[1] + bounding_box[3]) / 2
    Area = [1] + Area # 添加一个虚拟id，为了使用whether函数
    # 中心点是否落在Area内
    return whether_in_area((x_center, y_center), Area)

def speed_extract(last_traj, now_traj):
    """
    :param last_traj: 若干秒前的轨迹数据
    :param now_traj: 当前时刻轨迹数据
    :return: 【水平速度， 垂直速度】
    """
    last_x = int(last_traj.loc['x'])
    last_y = int(last_traj.loc['y'])
    cur_x = int(now_traj.loc['x'])
    cur_y = int(now_traj.loc['y'])
    x_speed = (cur_x - last_x) / (int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    y_speed = (cur_y - last_y) / (int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    return [x_speed, y_speed]

def whether_in_area(point, bbox):
    """
    :param point: [x, y]
    :param bbox: [id,x1,y1,x2,y2]
    """
    if point[0] <= bbox[3] and point[0] >= bbox[1] and point[1] <= bbox[4] and point[1] >= bbox[2]:
        return 1
    else:
        return 0

def overlap(box1, box2, val):
    # 判断两个矩形是否相交
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/2993875.html
    # 然后把思路写成了代码
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return 0
    else:
        max_x1 = max(minx1, minx2) # x1的最大值
        min_x2 = min(maxx1, maxx2) # x2的最小值
        max_y1 = max(miny1, miny2) # y1的最大值
        min_y2 = min(maxy1, maxy2)  # y2的最小值
        Cross_area = (min_x2 - max_x1) * (min_y2 - max_y1)
        box1_area = (maxx1 - minx1) * (maxy1 - miny1)
        box2_area = (maxx2 - minx2) * (maxy2 - miny2)
        if Cross_area / box1_area > val or Cross_area / box2_area > val:
            return 1
        else:
            return 0
def whether_occlusion(bbox, cur_bbox_list, val):
    """
    :param bbox: [id,x1,y1,x2,y2]
    :param cur_bbox_list: [bbox1, bbox2,...]
    :param matched_id_list: [id1, id2,...]
    :return: flag, OAR
    """
    occlusion_bbox_list = []
    occlusion_id_list = []
    for i in range(len(cur_bbox_list)):
        # 判断这个bbox与剩下的bbox是否有遮挡
        flag = overlap(bbox[1:], cur_bbox_list[i][1:], val)
        if flag:
            if len(occlusion_id_list) == 0:
                occlusion_id_list.append(bbox[0])
                occlusion_bbox_list.append(bbox[1:])
            occlusion_bbox_list.append(cur_bbox_list[i][1:])
            occlusion_id_list.append(cur_bbox_list[i][0])
            break
    return occlusion_bbox_list, occlusion_id_list

def whether_in_OAR(point, OAR_list):
    flag = 0
    for oar in OAR_list:
        oar_id = [0, oar[0], oar[1], oar[2], oar[3]]
        if whether_in_area(point, oar_id):
            flag = whether_in_area(point, oar_id)
            break
    return flag


def OAR_extractor(his_traj_dataframe_list,val):
    # 1. 初始化遮挡区域和id列表
    OAR_list = []
    OAR_id_list = []
    # 2. 如果是第一帧则不做处理
    if len(his_traj_dataframe_list) == 0:
        return OAR_list, OAR_id_list
    # 3. 提取上一时刻的跟踪结果
    his_id_list = his_traj_dataframe_list[-1]['ID'].unique()
    his_bbox_list = []
    for i in range(len(his_id_list)):
        visual_traj = his_traj_dataframe_list[-1].iloc[i]
        his_bbox_list.append([visual_traj['ID'], visual_traj['x1'], visual_traj['y1'], visual_traj['x2'],
                              visual_traj['y2']])
    # 提取有历史纪录的、存在遮挡的船舶检测框及对应id，表示当前遮挡区域
    for i in range(len(his_bbox_list)):
        if i < len(his_bbox_list) - 1:
            occlusion_boxes, occlusion_ids = whether_occlusion(his_bbox_list[i], his_bbox_list[i + 1:], val)
            for index in range(len(occlusion_boxes)):
                if (occlusion_ids[index] not in OAR_id_list) and (occlusion_ids[index] in his_id_list):
                    OAR_list.append(occlusion_boxes[index])
                    OAR_id_list.append(occlusion_ids[index])
    return OAR_list, OAR_id_list

def motion_features_extraction(his_traj_dataframe_list, VIS_tra_cur):
    """
    :param his_traj_dataframe_list: 过去五秒内每秒的视觉轨迹数据
    :param VIS_tra_cur: 当前秒的视觉轨迹数据
    :return:
    """
    speed_list = []
    VIS_traj_cur_withfeature = VIS_tra_cur.copy()
    cur_id_list = VIS_tra_cur['ID'].unique()
    for i in range(len(cur_id_list)):
        speed_list.append('[0, 0]')
    VIS_traj_cur_withfeature['speed'] = speed_list
    for k in range(len(cur_id_list)):
        if len(his_traj_dataframe_list) == 0:
            #VIS_tra_cur_withfeature.iloc[k].loc['speed'] = [0, 0]
            continue
        id = cur_id_list[k]
        for i in his_traj_dataframe_list:
            his_id_list = list(i['ID'].unique())
            if id not in his_id_list:
                #VIS_tra_cur_withfeature.iloc[k].loc['speed'] = [0, 0]
                continue
            else:
                index = his_id_list.index(id)
                last_traj = i.iloc[index]
                VIS_traj_cur_withfeature.loc[k, 'speed'] = str(speed_extract(last_traj, VIS_traj_cur_withfeature.iloc[k]))
                break
    return VIS_traj_cur_withfeature

# 判断某一ID是否在过去五秒内的视觉轨迹内存在
def id_whether_stable(id, last_5_trajs):
    for traj in last_5_trajs:
        if id in list(traj['ID'].unique()):
            continue
        else:
            return False
    return True

# 目标检测跟踪
class VISPRO(object):
    def __init__(self, anti, val, t):
        self.anti = anti
        self.last5_vis_tra_list = []
        self.Vis_tra_cur_3      = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.Vis_tra_cur        = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.Vis_tra            = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.VIS_tra_last = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y', 'speed','timestamp'])
        self.OAR_list = []
        self.OAR_ids_list = []
        self.OAR_mmsi_list = []
        self.val = val
        self.t = t
        self.Anti_occlusion_traj = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','speed','timestamp'])

    def detection(self, image):
        # 用于目标检测
        im0 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        im0 = Image.fromarray(im0)
        bboxes = yolo.detect_image(im0)
        return bboxes

    def track(self, image, bboxes, bboxes_anti_occ, id_list, timestamp):
        """
        bboxes: 检测结果（不含遮挡区域）
        bboxes_anti_occ: 遮挡区域抗遮挡推算
        id_list: 抗遮挡推算对应ID
        """
        # 用于目标跟踪
        bbox_xywh, confs = [], []
        bbox_xywh_anti_occ, confs_anti_occ = [], []
        if len(bboxes) or len(bboxes_anti_occ):
            # 检测结果整理
            for x1, y1, x2, y2, _, conf in bboxes:
                #获取框信息 [中心点x坐标，中心点y坐标，宽度，高度]
                obj = [int((x1+x2)/2), int((y1+y2)/2),x2-x1, y2-y1]
                bbox_xywh.append(obj)#框信息
                confs.append(conf)#置信度
            # 抗遮挡预测结果整理
            for x1, y1, x2, y2, _, conf in bboxes_anti_occ:
                #获取框信息 [中心点x坐标，中心点y坐标，宽度，高度]
                obj = [int((x1+x2)/2), int((y1+y2)/2),x2-x1, y2-y1]
                bbox_xywh_anti_occ.append(obj)#框信息
                confs_anti_occ.append(conf)#置信度

            # 检测结果、抗遮挡预测结果转tensor
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            xywhs_anti_occ = torch.Tensor(bbox_xywh_anti_occ)
            confss_anti_occ = torch.Tensor(confs_anti_occ)
            # 放入DeepSORT, 输出outputs = [x1,y1,x2,y2,[track],ID]
            outputs = deepsort.update(xywhs, confss, image, xywhs_anti_occ, confss_anti_occ, id_list, timestamp)
            for value in list(outputs):
                x1, y1, x2, y2, _, track_id = value
                if track_id in id_list:
                    x1, y1, x2, y2, _, _ = bboxes_anti_occ[id_list.index(track_id)] # 要把没有历史纪录的id从id——list中删掉
                # 存储至pd中[ID,x1,y1,x2,y2,trackx,tracky,time]
                self.Vis_tra_cur_3 = self.Vis_tra_cur_3.append({'ID':track_id,\
                    'x1':int(x1),'y1':int(y1),'x2':int(x2),'y2':int(y2),'x':int((x1 + x2) / 2),\
                        'y':int((y1 + y2) / 2), 'timestamp':timestamp//1000}, ignore_index=True)

    def update_tra(self, Vis_tra, timestamp):
        # 用于轨迹更新
        self.Vis_tra_cur = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        id_list = self.Vis_tra_cur_3['ID'].unique()
        for k in range(len(id_list)):
            id_current = self.Vis_tra_cur_3[self.Vis_tra_cur_3['ID'] == id_list[k]].reset_index(drop=True)
            # 求取均值
            df = id_current.mean().astype(int)
            df['timestamp'] = timestamp // 1000
            self.Vis_tra_cur = self.Vis_tra_cur.append(df, ignore_index=True)
        self.Vis_tra_cur_3 = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])

        Vis_tra_cur_withfeature = motion_features_extraction(self.last5_vis_tra_list, VIS_tra_cur= self.Vis_tra_cur)
        self.Vis_tra = self.Vis_tra.append(Vis_tra_cur_withfeature)
        if len(self.last5_vis_tra_list) > 4:
            self.last5_vis_tra_list.pop(0)
        self.last5_vis_tra_list.append(Vis_tra_cur_withfeature)
        # 删除时间过长的数据  时间以2分钟为限
        time_limited = 2
        self.Vis_tra = self.Vis_tra.drop(self.Vis_tra[self.Vis_tra['timestamp'] <\
                                                      (timestamp // 1000 - time_limited * 60)].index)
        return Vis_tra_cur_withfeature

    def traj_prediction_via_visual(self, last_traj, timestamp, speed):
        """
        :param last_traj: 若干秒的轨迹
        :return:
        """
        Vis_tra_prediction = last_traj.copy()
        x_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[0])
        y_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[1])
        Vis_tra_prediction.loc['x'] = Vis_tra_prediction.loc['x'] + x_move
        Vis_tra_prediction.loc['x1'] = Vis_tra_prediction.loc['x1'] + x_move
        Vis_tra_prediction.loc['x2'] = Vis_tra_prediction.loc['x2'] + x_move
        Vis_tra_prediction.loc['y'] = Vis_tra_prediction.loc['y'] + y_move
        Vis_tra_prediction.loc['y1'] = Vis_tra_prediction.loc['y1'] + y_move
        Vis_tra_prediction.loc['y2'] = Vis_tra_prediction.loc['y2'] + y_move
        Vis_tra_prediction.loc['timestamp'] = timestamp

        return Vis_tra_prediction

    def anti_occ(self, last5_vis_tra_list, bboxes, AIS_vis, bind_inf,timestamp):
        # 1.参数初始化
        bboxes_anti_occ = []
        if len(self.OAR_list):
            # 2. 删除处在OAR内的检测结果
            pop_index_list = []
            for index in range(len(bboxes)):
                for OAR in self.OAR_list:
                    if box_whether_in_area(bboxes[index][:4], OAR):
                        pop_index_list.append(index)
                        break
            for pop_index in range(len(pop_index_list)):
                bboxes.pop(pop_index_list[pop_index] - pop_index)
            
            # 所有遮挡id的mmsi提取
            bind_id_list = list(bind_inf['ID'].unique())
            self.OAR_mmsi_list = []
            OAR_ids_list_copy = self.OAR_ids_list.copy()
            for k in range(len(OAR_ids_list_copy)):
                if OAR_ids_list_copy[k] in bind_id_list:
                    mmsi = bind_inf.iloc[bind_id_list.index(OAR_ids_list_copy[k])].loc['mmsi']
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], int(mmsi)])
                else:
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], 0])

            # 预测bbox位置
            ais_vis_mmsi_list = list(AIS_vis['mmsi'])
            pop_index_list = []
            for k in range(len(self.OAR_mmsi_list)):
                final_find_flg = 0 # 是否找到最后时刻的位置
                second_final_find_flg = 0 # 是否找到前一时刻的位置
                final_pos = [] # 最新时刻的AIS投影位置
                second_final_pos = [] # 上一时刻的AIS投影位置
                # 若存在MMSI
                if not self.OAR_mmsi_list[k][1] == 0 and self.OAR_mmsi_list[k][1] in ais_vis_mmsi_list:
                    for i in range(len(ais_vis_mmsi_list)):
                        # 找到这条mmsi最后位置
                        # 对于有MMSI的船，用AIS预测位置
                        if int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][1] and \
                                int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 1:
                            final_find_flg = 1
                            final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                         AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        # 找到这条mmsi前一个位置
                        elif int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][
                            1] and int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 2:
                            second_final_find_flg = 1
                            second_final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                                AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        # 位置作差，得出位移向量，作用在视觉目标上
                        if final_find_flg and second_final_find_flg:
                            x_motion = final_pos[0] - second_final_pos[0]
                            y_motion = final_pos[1] - second_final_pos[1]
                            # 预测结果添加到抗遮挡检测框中
                            bboxes_anti_occ.append(
                                (self.Anti_occlusion_traj.iloc[k].loc['x1'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y1'] + y_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['x2'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y2'] + y_motion,
                                 'vessel', 1))  # 预测框生成
                            break
                # 若不存在MMSI
                else:
                    # 对于没有MMSI的船，用视觉预测位置
                    # 找到n秒前的轨迹点
                    # 如果过去五秒内轨迹不稳定，不预测
                    if not id_whether_stable(self.OAR_mmsi_list[k][0], last5_vis_tra_list):
                        pop_index_list.append(k)
                        continue
                    index = list(last5_vis_tra_list[0]['ID'].unique()).index(self.OAR_mmsi_list[k][0])
                    # 获取轨迹速度
                    speed_str = last5_vis_tra_list[0].iloc[index].loc['speed']
                    speed = [float(speed_str[1:-1].split(',')[0]), float(speed_str[1:-1].split(',')[1])]
                    # 轨迹速度作用在n秒前的轨迹点上
                    trajs = last5_vis_tra_list[0]
                    id_list = list(trajs['ID'].unique())
                    last_traj = trajs.iloc[id_list.index(self.OAR_mmsi_list[k][0])]
                    Vis_traj_now = self.traj_prediction_via_visual(last_traj, timestamp, speed)
                    # 添加到抗遮挡检测结果中
                    bboxes_anti_occ.append(
                        (Vis_traj_now.loc['x1'],
                         Vis_traj_now.loc['y1'],
                         Vis_traj_now.loc['x2'],
                         Vis_traj_now.loc['y2'],
                         'vessel', 1))

                # 删除既没有五秒历史数据，又没有AIS的目标
            for i in range(len(pop_index_list)):
                self.OAR_mmsi_list.pop(pop_index_list[i] - i)
                self.OAR_ids_list.pop(pop_index_list[i] - i)
                self.OAR_list.pop(pop_index_list[i] - i)
            if not len(self.OAR_ids_list) == len(bboxes_anti_occ):
                embed()
        return bboxes_anti_occ

        
    def feedCap(self, image, timestamp, AIS_vis, bind_inf):
        # 情况1: 当前时刻需要进行检测
        if timestamp % 1000 < self.t:
            
            # 1.1.目标检测框生成
            bboxes = self.detection(image)
            # print(bboxes)
            # 1.2.抗遮挡
            bboxes_anti_occ = self.anti_occ(self.last5_vis_tra_list, bboxes, AIS_vis, bind_inf, timestamp // 1000)

            # 1.3.DeepSORT跟踪
            # print(bboxes_anti_occ)
            self.track(image, bboxes, bboxes_anti_occ=bboxes_anti_occ,\
                    id_list=self.OAR_ids_list, timestamp=timestamp // 1000)

            # 轨迹数据更新
            Vis_tra_cur = self.Vis_tra_cur
            if timestamp % 1000 < self.t:
                Vis_tra_cur = self.update_tra(self.Vis_tra, timestamp)
                if self.anti:
                    # 根据上一时刻跟踪结果，提取出存在AIS的遮挡重叠船舶框以及对应ID
                    self.OAR_list, self.OAR_ids_list = OAR_extractor(self.last5_vis_tra_list, self.val)
                    # print("OAR_id_list", self.OAR_ids_list)
                self.VIS_tra_last = Vis_tra_cur

                # 更新被遮挡id对应的轨迹数据
                self.Anti_occlusion_traj = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
                id_list = list(self.VIS_tra_last['ID'].unique())
                for i in self.OAR_ids_list:
                    self.Anti_occlusion_traj = self.Anti_occlusion_traj.append(self.VIS_tra_last.iloc[id_list.index(i)])
        return self.Vis_tra, self.Vis_tra_cur
