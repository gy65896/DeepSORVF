import os, time, imutils, cv2, argparse
import pandas as pd
import numpy as np
from utils.file_read import read_all, ais_initial, update_time, time2stamp

from utils.VIS_utils import VISPRO
from utils.AIS_utils import AISPRO
from utils.FUS_utils import FUSPRO
from utils.gen_result import gen_result
import imageio
from utils.draw import DRAW


def main(arg):
    # 输出ais文件列表，从第几个开始数，时间戳，时间
    ais_file, timestamp0, time0 = ais_initial(arg.ais_path, arg.initial_time)
    Time = arg.initial_time.copy()
    # =============================================================================
    # 其他参数初始化
    # cap: 视频读取
    # fps: 帧率
    # t: 每帧持续时间
    # AIS: AIS处理初始化
    # VIS: VIS处理初始化
    # FUS: FUS处理初始化
    # name: 显示视频框的名称
    # show_size: 视频展示时的图像尺寸
    # videoWriter: 保存视频（None）
    # time_long: 视频运行时间
    # bin_inf: 绑定数据
    # =============================================================================
    
    cap = cv2.VideoCapture(arg.video_path)
    im_shape = [cap.get(3), cap.get(4)]
    max_dis = min(im_shape)//2
    fps = int(cap.get(5))
    t = int(1000/fps)
    
    AIS = AISPRO(arg.ais_path, ais_file, im_shape, t) # ais path, ais file, im_shape, t
    VIS = VISPRO(arg.anti, arg.anti_rate, t) # anti-occlusion, occlusion rate, t
    FUS = FUSPRO(max_dis, im_shape, t) # max distance of matching, im_shape, t
    DRA = DRAW(im_shape, t) # im_shape, t
    
    name = 'demo'
    show_size = 500
    videoWriter = None
    bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    # =============================================================================
    #  视频读取
    # =============================================================================
    print('Start Time: %s || Stamp: %d || fps: %d' % (time0, timestamp0, fps))
    times  = 0
    time_i = 0
    sum_t  = []

    while True:
        # 逐帧读取
        _, im = cap.read()
        if im is None:
            break
        start = time.time()
        
        # 更新时间戳
        Time, timestamp, Time_name = update_time(Time, t)
        # =============================================================================
        #  1.AIS轨迹提取, 每一秒钟对AIS数据更新一次
        # =============================================================================
        AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
        # =============================================================================
        #  2.视觉轨迹提取
        # =============================================================================
        Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, bin_inf)
        # =============================================================================
        #  3.视觉AIS数据融合
        # =============================================================================
        Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)

        end = time.time() - start
        time_i = time_i + end
        if timestamp % 1000 < t:
            gen_result(times, Vis_cur, Fus_tra, arg.result_metric, im_shape)
            times = times+1
            sum_t.append(time_i)
            print('Time: %s || Stamp: %d || Process: %.6f || Average: %.6f +- %.6f'%(Time_name, timestamp, time_i, np.mean(sum_t), np.std(sum_t)))
            time_i = 0
        # =============================================================================
        #  4.融合结果显示
        # =============================================================================
        im = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)
        # =============================================================================
        #  视频展示
        # =============================================================================
        result = im
        result = imutils.resize(result, height=show_size)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                arg.result_video, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)

        cv2.imshow(name, result)
        cv2.waitKey(1)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        # 点x退出
            break   
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
        # =============================================================================
        #     该开关定义了所有需要修改的参数
        # =============================================================================
        parser = argparse.ArgumentParser(description = "DeepSORVF")
        
        parser.add_argument("--anti", type=int, default = 1, help='anti-occlusion True/1|False/0')
        parser.add_argument("--anti_rate", type=int, default = 0, help='occlusion rate 0-1')
        
        parser.add_argument("--data_path", type=str, default = './clip-01/', help='data path')
        parser.add_argument("--result_path", type=str, default = './result/', help='result path')
        
        video_path, ais_path, result_video, result_metric, initial_time,\
            camera_para = read_all(parser.parse_args().data_path, parser.parse_args().result_path)
    
        parser.add_argument("--video_path", type=str, default = video_path, help='video path')
        parser.add_argument("--ais_path", type=str, default = ais_path, help='ais path')
        parser.add_argument("--result_video", type=str, default = result_video, help='result video')
        parser.add_argument("--result_metric", type=str, default = result_metric, help='result metric')
        parser.add_argument("--initial_time", type=list, default = initial_time, help='initial time')
        parser.add_argument("--camera_para", type=list, default = camera_para, help='camera para')

        argspar = parser.parse_args()
    
        print("\nVesselSORT")
        for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
        arg = parser.parse_args()
    
        main(arg)
