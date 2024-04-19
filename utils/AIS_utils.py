import pandas as pd
from geopy.distance import geodesic
import pyproj
from math import radians, cos, sin, asin, sqrt, tan, atan2, degrees
import math
import numpy as np
import cv2
from IPython import embed
import os

def count_distance(point1, point2, Type='m'):
    '''
    功能: 使用经纬度计算两点间的距离，单位m
    point1: 点1的经纬度（经度，纬度）
    point2: 点2的经纬度（经度，纬度）
    Type: nm——海里; m——米
    返回值: 二者之间的距离，单位为米
    '''
    # 计算两点间距离，单位m
    distance = geodesic(point1, point2).m  # 计算两经纬度之间的距离
    if Type == 'nm':
        # 将距离单位转换为nm
        distance = distance * 0.00054
    return distance


def getDegree(latA, lonA, latB, lonB):
    '''
    功能: 计算两点间方位角
    latA: 相机纬度
    LonA: 相机经度
    latB: 船舶纬度
    LonB: 船舶经度
    返回值: 方位角
    '''
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng

def visual_transform(lon_v, lat_v, camera_para, shape):
    '''
    功能: 计算两点间方位角
    lon_cam: 相机经度（单位）
    lat_cam: 相机纬度
    shoot_vdir： 相机垂直朝向向下倾斜多少度
    shoot_hdir: 水平朝向
    height_cam：相机距离水面高度
    width_pic：图片宽
    height_pic：图片高
    FOV_hor：水平视场角 55
    FOV_ver：垂直视场角
    返回值: 方位角
    '''
    # 初始化
    lon_cam = camera_para[0]
    lat_cam = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    FOV_hor = camera_para[5]
    FOV_ver = camera_para[6]
    width_pic = shape[0]
    height_pic = shape[1]
    f_x = camera_para[7]
    f_y = camera_para[8]
    u0  = camera_para[9]
    v0  = camera_para[10]

    # 1.计算距离
    D_abs = count_distance((lat_cam, lon_cam), (lat_v, lon_v))
    
    # 2.计算水平夹角
    relative_angle = getDegree(lat_cam, lon_cam, lat_v, lon_v)
    Angle_hor = relative_angle - shoot_hdir
    if Angle_hor < -180:
        Angle_hor = Angle_hor + 360
    elif Angle_hor > 180:
        Angle_hor = Angle_hor - 360
    hor_rad = radians(Angle_hor)
    shv_rad = radians(-shoot_vdir)
    Z_w = D_abs*cos(hor_rad)
    X_w = D_abs*sin(hor_rad)
    Y_w = height_cam
    Z = Z_w/cos(shv_rad)+(Y_w-Z_w*tan(shv_rad))*sin(shv_rad)
    X = X_w
    Y = (Y_w-Z_w*tan(shv_rad))*cos(shv_rad)
    #print(X,Y,Z)
    target_x = int(f_x*X/Z+u0)
    target_y = int(f_y*Y/Z+v0)
    # 3.计算垂直夹角
    #Angle_ver = 90 + shoot_vdir - math.degrees(math.atan(D_abs / height_cam))
    #print(Angle_ver, shoot_vdir)
    # 4.计算坐标
    #target_x1 = int(width_pic // 2 + width_pic * Angle_hor / FOV_hor)
    #target_y1 = int(height_pic // 2 + height_pic * Angle_ver / FOV_ver)
    #print('new:')
    #print(target_x2, target_y2)
    #print('origion:')
    # print(Angle_hor)
    # print(target_x, target_y)
    return target_x, target_y

def data_filter(ais, camera_para):
    '''
    对推算后的数据进行筛选
    :param ais: 当前时刻的船舶数据
    :param lon_cam: 摄像头经度
    :param lat_cam: 摄像头纬度
    :param max_dis: 摄像头探测距离
    :return: 预处理后的当前时刻船舶数据
    '''
    # 初始化
    lon_cam = camera_para[0]
    lat_cam = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    FOV_hor = camera_para[5]
    FOV_ver = camera_para[6]

    lon, lat = ais['lon'], ais['lat']
    D_abs = count_distance((lat_cam,lon_cam),(lat,lon))
    angle = getDegree(lat_cam, lon_cam, lat, lon)
    in_angle = abs(shoot_hdir - angle) if abs(shoot_hdir -
                        angle) < 180 else 360 - abs(shoot_hdir - angle)

    # 先进行相机垂直方向可视范围的判断，若垂直方向上在可视范围内，则再进行水平方向的判断
    if 90 + shoot_vdir - FOV_ver / 2 < math.degrees(math.atan(D_abs / height_cam)):
        # =============================================================================
        #         坐标转换及视觉轨迹筛选范围
        # =============================================================================
        # 相机水平方向可视范围判断，若在范围内，则返回1代表可进行坐标转换
        # 此处我们在相机水平方向的可视范围基础上适当增加相应角度
        # 处理船舶AIS数据位置相对船舶视频位置靠后的问题，以提高边缘船舶融合效果
        if in_angle <= (FOV_hor / 2 + 8):  # 此处“+x”为相应增加的角度
            return 'transform'
        # 若当前AIS数据超出设定的扇形角度(可视角度+相应增加的角度)，则不进行坐标转换。
        # 返回0,删除掉超出该范围的视觉轨迹
        elif in_angle > (FOV_hor / 2 + 8):
            return 'visTraj_del'
        # =============================================================================
        #         AIS数据筛选范围
        # =============================================================================
        # 如果当前AIS数据在更大范围的扇形范围内，则将此AIS数据删掉，不做考虑
        if in_angle > (FOV_hor / 2 + 12):
            return 'ais_del'


def transform(AIS_current, AIS_vis, camera_para, shape):
    '''
    功能: 将AIS数据转换至图像坐标系
    :param AIS_current: 当前AIS数据
    :param AIS_vis: 含有图像坐标的AIS数据
    :return: 当前处理后的AIS数据转换到图像坐标系中的AIS数据
    '''
    # 1.数据初始化
    AIS_visCurrent = pd.DataFrame(columns=['mmsi','lon',\
                                'lat','speed','course','heading','type','x','y','timestamp'])
    # 2.遍历所有数据
    for index, ais in AIS_current.iterrows():
        # 判断数据是否可以进行转换
        flag = data_filter(ais, camera_para)
        # 情况1: 坐标转换
        if flag == 'transform':
            x, y = visual_transform(ais['lon'], ais['lat'], camera_para, shape)
            ais['x'], ais['y'] = x, y
            AIS_visCurrent = AIS_visCurrent.append(ais, ignore_index=True)
        # 情况2: 数据删除
        elif flag == 'visTraj_del' or flag == 'ais_del':
            AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['mmsi'] == ais['mmsi']].index)
    return AIS_vis, AIS_visCurrent


def data_pre(ais, timestamp):
    '''
    对船舶AIS数据进行推算，并添加在当前时刻AIS_current中
    :param ais: 上一秒未在当前时刻出现的船舶数据
    :param timestamp:时间戳
    :return: 修改后的AIS数据
    '''
    # 若船舶速度为0，则仅修改其时间，并添加加到当前时刻的船舶数据中
    if ais['speed'] == 0:
        ais['timestamp'] = timestamp
    # 否则对其进行推算，并添加到当前时刻的船舶数据中
    else:
        geo_d = pyproj.Geod(ellps="WGS84")
        
        # 算下一时刻距离
        distance = ais['speed'] * ((timestamp - ais['timestamp']) / 3600) * 1852
        ais['timestamp'] = timestamp

        # 算经纬度
        ais['lon'], ais['lat'], c = geo_d.fwd(
            ais['lon'], ais['lat'], ais['course'], distance)
    return ais

def data_pred(AIS_cur, AIS_read, AIS_las, timestamp):

    for index, ais in AIS_read.iterrows():
        ais['timestamp'] = round(ais['timestamp']/1000)
        # 1.当前时刻存在不推算
        if ais['timestamp'] == int(timestamp//1000):
            AIS_cur = AIS_cur.append(ais, ignore_index=True)
        # 2.当前时刻不存在推算
        else:
            AIS_cur = AIS_cur.append(data_pre(ais,\
                            timestamp//1000), ignore_index=True)
    
    for index, ais in AIS_las.iterrows():
        if ais['mmsi'] not in AIS_cur['mmsi'].values:
            AIS_cur = AIS_cur.append(data_pre(ais,\
                        timestamp//1000), ignore_index=True)
    return AIS_cur

def data_coarse_process(AIS_current,AIS_last,camera_para,max_dis):
    '''
    数据预处理，包括数据清洗、数据筛选
    :param AIS_current: 当前时刻的船舶数据
    :param AIS_last:上一时刻的船舶数据
    :param lon_cam: 摄像头经度
    :param lat_cam: 摄像头纬度
    :param max_dis: 摄像头探测距离(m)
    :return: 预处理后的当前时刻船舶数据
    '''
    camera_loc = (camera_para[1], camera_para[0])
    
    for index, ais in AIS_current.iterrows():
        flag = 0
        # 1.清洗异常数据
        if ais['mmsi'] / 100000000 < 1 or ais['mmsi'] / 100000000 >= 10 or\
            ais['lon'] == -1 or ais['lat'] == -1 or ais['speed'] == -1 or\
                ais['course'] == -1 or ais['course'] == 360 or ais['heading'] == -1 or ais['lon'] > 180 or\
                    ais['lon'] < 0 or ais['lat'] > 90 or ais['lat'] < 0 or ais['speed'] <= 0.3:
            AIS_current = AIS_current.drop(index=index)
            continue

        # 2.清洗经纬度速度发生较大变化的数据
        # 在此之前需考虑当前时刻的船舶数据是否会出现在上一时刻中，若出现则求取变化值，若未出现则不进行操作
        if ais['mmsi'] in AIS_last['mmsi'].values:
            temp = AIS_last[AIS_last.mmsi == ais['mmsi']]
            if abs(ais['lon'] - temp['lon'].values[-1]) >= 1 \
                or abs(ais['lat'] - temp['lat'].values[-1]) >= 1 \
                    or abs(ais['speed'] - temp['speed'].values[-1]) >= 7:
                AIS_current = AIS_current.drop(index=index)
                continue
   
        # 3.清洗距离过远或特定区域以外的数据
        ship_loc = (ais['lat'], ais['lon'])
        dis = count_distance(camera_loc, ship_loc, Type='m')
        if dis > max_dis or data_filter(ais, camera_para) == 'ais_del':
            AIS_current = AIS_current.drop(index=index)
    return AIS_current


class AISPRO(object):
    def __init__(self, ais_path, ais_file, im_shape, t):
        # ais路径
        self.ais_path = ais_path
        # ais文件名list
        self.ais_file = ais_file
        # 图像尺寸
        self.im_shape = im_shape
        # 保留ais数据的最大距离
        self.max_dis  = 2*1852
        # 每帧图像显示时间
        self.t        = t
        # 数据保存时长
        self.time_lim = 2
        # 数据1: 当前时刻AIS数据
        self.AIS_cur  = pd.DataFrame(columns=['mmsi','lon','lat','speed',\
                                              'course','heading','type','timestamp'])
        # 数据2: 原始AIS数据
        # self.AIS_row  = pd.DataFrame(columns=['mmsi','lon','lat','speed',\
        #                                       'course','heading','time'])
        # 数据3: 推算AIS数据
        # self.AIS_pre  = pd.DataFrame(columns=['mmsi','lon','lat','speed',\
                                              # 'course','heading','time'])
        # 数据4: 坐标转换的AIS数据
        self.AIS_vis  = pd.DataFrame(columns=['mmsi','lon','lat','speed',\
                                              'course','heading','type','x','y','timestamp'])
    def initialization(self):
        # 参数初始化
        AIS_las = self.AIS_cur
        AIS_vis = self.AIS_vis
        AIS_cur = pd.DataFrame(columns=['mmsi','lon',\
                                    'lat','speed','course','heading','type','timestamp'])
        return AIS_cur, AIS_las, AIS_vis
    
    def read_ais(self, Time_name):
        try:
            # 读取ais数据
            path = self.ais_path + '/' + Time_name + '.csv'
            ais_data = pd.read_csv(path, usecols=[1, 2, 3, 4, 5, 6, 7, 8], header=0)
            # self.AIS_row = self.AIS_row.append(ais_data, ignore_index=True)
        except:
            ais_data = pd.DataFrame(columns=['mmsi','lon',\
                                    'lat','speed','course','heading','type','timestamp'])
        return ais_data
    
    def data_tran(self, AIS_cur, AIS_vis, camera_para, timestamp):
        # 1.AIS数据坐标转换
        AIS_vis, AIS_vis_cur = transform(AIS_cur,\
                            AIS_vis, camera_para, self.im_shape)

        # 2.存储处理后的AIS数据
        # self.AIS_pre = self.AIS_pre.append(self.AIS_cur, ignore_index=True)
        AIS_vis = AIS_vis.append(AIS_vis_cur, ignore_index=True)
        
        # 3.删除时间过长的AIS数据  时间以3分钟为限
        # self.AIS_pre = self.AIS_pre.drop(self.AIS_pre[self.AIS_pre['time'] < (
        #     timestamp // 1000 - self.time_lim * 60)].index)
        AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['timestamp'] < (
                timestamp//1000 - self.time_lim * 60)].index)
        return AIS_vis
    
    def ais_pro(self, AIS_cur, AIS_las, AIS_vis, camera_para, timestamp, Time_name):

        # 1.读取文件
        AIS_read = self.read_ais(Time_name)
                
        # 2.数据粗清洗
        AIS_read = data_coarse_process(AIS_read, AIS_las,\
                                      camera_para, self.max_dis)
                
        # 3.对未出现的AIS数据推算
        AIS_cur = data_pred(AIS_cur, AIS_read, AIS_las, timestamp)
            
        # 5.坐标转换
        AIS_vis = self.data_tran(AIS_cur, AIS_vis,\
                                      camera_para, timestamp)
        return AIS_vis, AIS_cur
    
    def process(self, camera_para, timestamp, Time_name):
        # 当前时刻需要进行更新
        if timestamp % 1000 < self.t:
            Time_name = Time_name[:-4]
            # 1.参数初始化
            AIS_cur, AIS_las, AIS_vis = self.initialization()
            
            # 2.数据生成
            self.AIS_vis, self.AIS_cur = self.ais_pro(AIS_cur,\
                                AIS_las, AIS_vis, camera_para, timestamp, Time_name)
            
        return self.AIS_vis, self.AIS_cur

