import pandas as pd

def gen_result(frame,vis,fus,result_name,im_shape):

    for index, inf in vis.iterrows():
        ID = int(inf['ID'])
        x    = max(int(inf['x1']),0)
        y    = max(int(inf['y1']),0)
        w    = min(int(inf['x2']),im_shape[0])-x
        h    = min(int(inf['y2']),im_shape[1])-y
        detection = [[frame,0,x,y,w,h,1,1,1,1]]
        tracking = [[frame,ID,x,y,w,h,1,1,1,1]]
        df_detection = pd.DataFrame(data=detection)
        df_detection.to_csv(result_name[:-4]+'_detection'+\
                result_name[-4:], mode='a', index = False, header=False)
        df_tracking = pd.DataFrame(data=tracking)
        df_tracking.to_csv(result_name[:-4]+'_tracking'+\
                result_name[-4:], mode='a', index = False, header=False)
        
    for index, inf in fus.iterrows():
        mmsi = int(inf['mmsi'])
        x    = int(inf['x1'])
        y    = int(inf['y1'])
        w    = int(inf['w'])
        h    = int(inf['h'])
        s = [[frame,mmsi,x,y,w,h,1,1,1,1]]
        df = pd.DataFrame(data=s)
        df.to_csv(result_name[:-4]+'_fusion'+\
                 result_name[-4:], mode='a', index = False, header=False)
    