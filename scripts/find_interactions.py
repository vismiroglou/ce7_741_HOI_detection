import os
from glob import glob
import shutil

def find_interactions(root, destination):

    dir = os.path.join(root, 'annotations')
    for day in os.listdir(dir):
        for clip in os.listdir(os.path.join(dir, day)):
            clip_dir = os.path.join(dir, day, clip)
            for anno in glob(os.path.join(clip_dir, '*.txt')):
                with open(anno, 'r') as f:
                    text = f.read()
                if 'bicycle' in text or 'motorcycle' in text:
                    
                    clip_path = clip_dir.replace('annotations', 'clips') + '.mp4'
                    clip_destination = os.path.join(destination, 'clips', day, clip+'.mp4')

                    anno_path = clip_dir
                    anno_destination = os.path.join(destination, 'annotations', day, clip)

                    print(anno_destination)
                    print(clip_destination)

                    # if not os.path.isdir(clip_destination):
                    #     os.makedirs(clip_destination)
                    
                    # if not os.path.isdir(anno_destination):
                    #     os.makedirs(anno_destination)
                    
                   
                    # try:
                    #     shutil.move(clip_path, clip_destination)
                    # except Exception as e: print(e)

                    # try:
                    #     shutil.move(anno_path, anno_destination)
                    # except Exception as e: print(e)

                    break