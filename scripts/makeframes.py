import cv2 as cv
import os

def make_frames(clip_dir):
    pathOut = clip_dir.replace('.mp4','')
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    vidcap = cv.VideoCapture(clip_dir)
    count = 1
    success = True
    while success:
        success,image = vidcap.read()
        if count%25 == 0:
            frame_path = os.path.join(pathOut, 'frame_%s.jpg'%'{0:04}'.format(int(count/25)))
            anno_path = frame_path.replace('clips', 'annotations').replace('frame', 'annotations').replace('.jpg', '.txt')
            if os.path.exists(anno_path):
                cv.imwrite(frame_path, image)
            else:
                print('Annotation for', frame_path, 'not found')
        count+=1


if __name__ == '__main__':
    #Extracts frames for all clips 
    dir = '../../data/clips'
    for folder in os.listdir(dir):
        day = os.path.join(dir, folder)
        for clip in os.listdir(day):
            clip_path = os.path.join(day, clip)
            make_frames(clip_path)