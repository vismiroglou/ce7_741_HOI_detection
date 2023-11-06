import cv2 as cv
import os
from glob import glob

def make_frames(clip_dir):
    pathOut = clip_dir.replace('.mp4','')
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    
        vidcap = cv.VideoCapture(clip_dir)
        count = 0
        success = True
        while success:
            success,image = vidcap.read()
            if count%25 == 0 and count>24:
                frame_path = os.path.join(pathOut, 'frame_%s.jpg'%'{0:04}'.format(int((count/25)-1)))
                anno_path = frame_path.replace('clips', 'annotations').replace('frame', 'annotations').replace('.jpg', '.txt')
                if os.path.exists(anno_path):
                    cv.imwrite(frame_path, image)
                else:
                    print('Annotation for', frame_path, 'not found')
            count+=1
    else:
        print('Video', clip_dir, 'has already been extracted.')


if __name__ == '__main__':
    #Extracts frames for all clips 
    
    make_frames(r'..\data\mini_test_folder\clips\20200517\clip_21_0949.mp4')
    # for folder in os.listdir(dir):
    #     day = os.path.join(dir, folder)
    #     for clip in os.listdir(day):
    #         clip_path = os.path.join(day, clip)
    #         make_frames(clip_path)