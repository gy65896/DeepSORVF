import time
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from IPython import embed

def __reduce_by_half(x):
    # 轨迹压缩
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

def angle(v1, v2):
    # 计算轨迹速度的夹角
    if len(v1) >= 10:
        dx1 = v1[-1][0] - v1[-10][0]
        dy1 = v1[-1][1] - v1[-10][1]
    elif len(v1) < 10:
        dx1 = v1[-1][0] - v1[0][0]
        dy1 = v1[-1][1] - v1[0][1]
    if len(v2) >= 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
    elif len(v2) < 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
    
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > math.pi:
            included_angle = math.pi*2 - included_angle
    return included_angle

def DTW_fast(traj0, traj1):
    # 1.计算轨迹间夹角
    if len(traj0)>1 and len(traj1)>1:
        theta = angle(traj0, traj1)
        traj0 = __reduce_by_half(traj0)
        traj1 = __reduce_by_half(traj1)
    else:
        theta = 0
    
    # 2.使用fastDTW
    d, path = fastdtw(traj0, traj1, dist=euclidean)
    
    return d*math.exp(theta)


def traj_group(df_data, df_dataCur,  kind):
    """
    对数据轨迹按照MMSI呼号或者ID号进行分组，并获取每条船舶或者每个检测框的轨迹
    :param df_data: AIS数据或者视频数据
    :param kind: 数据类型：AIS或VIS
    返回分组之后的轨迹列表以及每条轨迹对应的MMSI呼号列表或者ID号列表，为匹配做准备
    :return: trajData_list, trajLabel_list, trajInf_list
    """
    # 1.初始化
    trajData_list = []  # 存储AIS位置信息(x,y)
    trajLabel_list = []  # 存储AIS的MMSI呼号
    trajInf_list = []  # 存储AIS所有信息
    
    # 2.AIS数据分组
    if kind == 'AIS':
        grouped = df_data.groupby('mmsi')
        for value, group in grouped:
            
            # 仅记录当前时刻存在的船
            if value in df_dataCur['mmsi'].tolist():
                traj = group.values
                
                trajData_list.append(np.array(traj[:, 7:9]))
                trajLabel_list.append(int(traj[0, 0]))
                trajInf_list.append(traj)
    
    # 3.VIS数据分组
    elif kind == 'VIS':
        grouped = df_data.groupby('ID')
        for value, group in grouped:
            
            # 仅记录当前时刻存在的船
            if value in df_dataCur['ID'].tolist():
                traj = group.values
                
                trajData_list.append(np.array(traj[:, 5:7]))
                trajLabel_list.append(int(traj[0][0]))
                trajInf_list.append(traj)

    return trajData_list, trajLabel_list, trajInf_list

class FUSPRO(object):
    def __init__(self, max_dis, im_shape, t):
        # 最大匹配距离
        self.max_dis = max_dis
        self.im_shape = im_shape
        # 绑定次数阈值
        self.bin_num = 3
        # 遗忘阈值
        self.fog_num = 3
        # 每帧显示时间
        self.t = t

        # 数据1: 当前时刻的匹配数据
        self.mat_cur  = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi','timestamp', 'match']))
        # 数据2: 当前时刻的匹配信息
        self.mat_list = pd.DataFrame(columns=['ID', 'mmsi',\
                                'lon', 'lat', 'speed', 'course', 'heading', 'type', 'timestamp'])
        # 数据3: 当前时刻的绑定信息
        self.bin_cur  = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    def initialization(self, AIS_list, VIS_list):
        # 参数初始化
        mat_las   = self.mat_cur
        bin_las   = mat_las[mat_las['match'] > self.bin_num]
        mat_cur   = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi','timestamp', 'match']))
        bin_cur   = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        
        mat_list  = pd.DataFrame(columns=['ID', 'mmsi',\
                                'lon', 'lat', 'speed', 'course', 'heading', 'type', 'x1',\
                                    'y1', 'w', 'h', 'timestamp'])
        
        return mat_cur, bin_cur, mat_las, bin_las, mat_list
    
    def cal_similarity(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las):
        # 1.初始化
        matrix_S = np.zeros((len(VIS_list), len(AIS_list)))
        # 2.提取绑定信息
        binIDmmsi, bin_MMSI, bin_ID = [], [], []
        if len(bin_las)!=0:
            grouped = bin_las.groupby('ID/mmsi')
            for value, group in grouped:
                ID, MMSI = value.split('/')
                bin_ID.append(int(ID))
                bin_MMSI.append(int(MMSI))
                binIDmmsi.append(value)
                
        for i in range(len(VIS_list)):
            for j in range(len(AIS_list)):
                
                # 3.提取当前ID和mmsi
                cur_ID, cur_mmsi = VIS_IDlist[i], AIS_MMSIlist[j]
                cur_IDmmsi = str(int(cur_ID))+'/'+str(int(cur_mmsi))
                
                # 情况1: 绑定信息未提到时，FastDTW
                if int(cur_mmsi) not in bin_MMSI and int(cur_ID) not in bin_ID:
                    theta = angle(VIS_list[i], AIS_list[j])
                    # 计算距离
                    x_VIS = VIS_list[i][-1][0]
                    y_VIS = VIS_list[i][-1][1]
                    x_AIS = AIS_list[j][-1][0]
                    y_AIS = AIS_list[j][-1][1]
                    dis   = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
                    # 判断是否保存
                    if dis < self.max_dis and theta < math.pi*(5/6):
                        matrix_S[i][j] = DTW_fast(VIS_list[i], AIS_list[j])
                    else:
                        matrix_S[i][j] = 1000000000
                
                # 情况2: 存在绑定信息时，极小值
                elif cur_IDmmsi in binIDmmsi:
                    matrix_S[i][j] = 0-int(bin_las[bin_las['ID/mmsi'] == cur_IDmmsi]['match'].values)*100
                
                # 情况3: 存在绑定信息中但不是绑定值，无穷大
                else:
                    matrix_S[i][j] = 1000000000
        return matrix_S
    
    def data_filter(self, row_ind, col_ind, VIS_list, AIS_list):
        # 1.初始化
        matches = []

        # 2.删除过远或角度过大数据
        for row, col in zip(row_ind, col_ind):
            # 计算夹角
            theta = angle(VIS_list[row], AIS_list[col])
            
            # 计算距离
            x_VIS = VIS_list[row][-1][0]
            y_VIS = VIS_list[row][-1][1]
            x_AIS = AIS_list[col][-1][0]
            y_AIS = AIS_list[col][-1][1]
            dis   = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
            # 判断是否保存
            if dis < self.max_dis and theta < math.pi*(5/6): # 
                matches.append((row, col))
        return matches
    
    def save_data(self, mat_cur, bin_cur, mat_las, bin_las, mat_list,\
                  matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp):
        # 1.存储当前时刻匹配信息
        for i in range(len(matches)):
            v_loc, a_loc = matches[i][0],matches[i][1]
            ID           = int(VIS_IDlist[v_loc])
            MMSI         = int(AIS_MMSIlist[a_loc])
            ID_MMSI      = str(ID)+'/'+str(MMSI)

            lon          = AInf_list[a_loc][-1][1]
            lat          = AInf_list[a_loc][-1][2]
            speed        = AInf_list[a_loc][-1][3]
            course       = AInf_list[a_loc][-1][4]
            heading      = AInf_list[a_loc][-1][5]
            types        = AInf_list[a_loc][-1][6]
            time         = AInf_list[a_loc][-1][9]
            
            x1           = max(VInf_list[v_loc][-1][1],0)
            y1           = max(VInf_list[v_loc][-1][2],0)
            x2           = min(VInf_list[v_loc][-1][3],self.im_shape[0])
            y2           = min(VInf_list[v_loc][-1][4],self.im_shape[1])
            w            = abs(x2-x1)
            h            = abs(y2-y1)
            
            mat_list = mat_list.append({'ID':ID,'mmsi':MMSI,'lon':lon,'lat':lat,\
                'speed':speed,'course': course,'heading':heading,'type':types,'x1':x1,'y1':y1,\
                    'w':w,'h':h,'timestamp':time}, ignore_index=True)
            # 情况1: 历史存在该匹配对，match=match+1
            if ID_MMSI in mat_las['ID/mmsi'].values:
                match = mat_las[mat_las['ID/mmsi'] == ID_MMSI]['match'].values[0]+1
                mat_cur  = mat_cur.append({'ID/mmsi':str(ID)+'/'+str(MMSI),\
                                           'timestamp':time,'match':match}, ignore_index=True)
            # 情况2: 历史不存在该匹配对，match=1
            else:  
                mat_cur  = mat_cur.append({'ID/mmsi':str(ID)+'/'+str(MMSI),\
                                           'timestamp':time,'match':1}, ignore_index=True)
        
        # 情况3: 历史存在匹配对当前不存在(不是由于MMSI走出画面，且在特定时间内)，match=match
        for ind, inf in bin_las.iterrows():
            ID_MMSI = inf['ID/mmsi']
            ID, MMSI = [int(x) for x in ID_MMSI.split('/')]
            time    = inf['timestamp']
            if MMSI in AIS_MMSIlist and ID_MMSI not in mat_cur['ID/mmsi'].values\
                                                and timestamp//1000-time < self.fog_num:
                mat_cur = mat_cur.append(inf, ignore_index=True)
        
        # 2.存储当前时刻绑定信息
        for ind, inf in mat_cur.iterrows():
            ID, MMSI = [int(x) for x in inf['ID/mmsi'].split('/')]
            if inf['match'] > self.bin_num:
                bin_cur = bin_cur.append({'ID': ID, 'mmsi': MMSI,\
                  'timestamp': int(inf['timestamp']), 'match': int(inf['match'])}, ignore_index=True)

        return mat_list, mat_cur, bin_cur
    
    def traj_match(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp):
        
        # 1.初始化
        mat_cur, bin_cur, mat_las, bin_las, mat_list = self.initialization(AIS_list, VIS_list)
        
        # 2.相似性度量
        matrix_S = self.cal_similarity(AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las)
        
        # 3.匈牙利最优匹配
        row_ind, col_ind = linear_assignment(matrix_S)
        # 4.数据滤波
        matches = self.data_filter(row_ind, col_ind, VIS_list, AIS_list)

        matric = pd.DataFrame(matrix_S,columns=AIS_MMSIlist,index=VIS_IDlist,dtype=int)
        # print(matric)

        # for row, col in zip(row_ind, col_ind):
            # 计算夹角
            # print(VIS_IDlist[row], AIS_MMSIlist[col])
        # 5.保存数据
        mat_list, mat_cur, bin_cur = self.save_data(mat_cur, bin_cur, mat_las, bin_las,\
                                    mat_list, matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return mat_list, mat_cur, bin_cur
    
    def fusion(self,AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp):
        if timestamp % 1000 < self.t:
            # 1.信息分组提取
            AIS_list, AIS_MMSIlist, AInf_list = traj_group(AIS_vis, AIS_cur, 'AIS')
            VIS_list, VIS_IDlist, VInf_list = traj_group(Vis_tra, Vis_cur, 'VIS')

            # 2.轨迹匹配
            self.mat_list, self.mat_cur, self.bin_cur = self.traj_match(AIS_list,\
                                AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return self.mat_list, self.bin_cur

        
