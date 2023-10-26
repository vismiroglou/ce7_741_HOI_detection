import cv2 as cv
import os

def make_frames(clip_dir):
    pathOut = clip_dir.replace('.mp4','')
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    vidcap = cv.VideoCapture(clip_dir)
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        # print('read a new frame:',success)
        if count%25 == 0 :
            cv.imwrite(pathOut + r'/' + 'frame_%s.jpg'%'{0:04}'.format(int(count/25)), image)
        count+=1