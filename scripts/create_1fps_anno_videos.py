import os
import pandas as pd
import cv2
from glob import glob
import re


def make_videos(video_path):
    size = (384, 288)
    
    frame_dir = video_path.replace('.mp4', '')
    video_name = video_path.split('\\')[-1]

    if os.path.isdir(frame_dir):
        img_array=[]
        for frame in glob(os.path.join(frame_dir, '*.jpg')):
            anno_path = frame.replace('clips', 'annotations').replace('frame', 'annotations').replace('.jpg', '.txt')
            img = cv2.imread(frame)
            #Try to read annotations for the frame
            try:
                data = pd.read_csv(anno_path, sep=' ', header=None)
                for i in data.index:
                    x1 = data.loc[i, 2]
                    y1 = data.loc[i, 3]
                    x2 = data.loc[i, 4]
                    y2 = data.loc[i, 5]
                    if data.loc[i, 1] == 'human':
                        color = (0,255,0)
                    elif data.loc[i, 1] == 'bicycle':
                        color = (255,0,0)
                    elif data.loc[i, 1] == 'motorcycle':
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 255)

                    img = cv2.rectangle(img,(x1,y1),(x2,y2), color, 1)
                    img = cv2.putText(img, str(data.loc[i, 0]), org=(x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color)
            except:
                print('No annotations')

            img = cv2.putText(img, str(frame), org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255))
            img_array.append(img)
       

        video_out = os.path.join(frame_dir, '%s.avi'%video_name.replace('.mp4', ''))
        if not os.path.exists(video_out):
            out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
            
            print('Video', '%s.avi'%video_name.replace('.mp4', ''), 'completed')
        else:
            print('Video', '%s.avi'%video_name.replace('.mp4', ''), 'already exists')

if __name__ == '__main__':

    for video_path in glob(os.path.join(r'..\..\data_inter\clips\20200515', '*.mp4')):
        make_videos(video_path)