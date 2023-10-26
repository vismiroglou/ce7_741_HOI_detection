import os
import shutil

#data_list as the path to the file
def extract_data(root, destination, data_list):
    with open(data_list, 'r') as f:
        data = f.read()
        data = data.split("\n").replace('../data', root)

    for i in data:
        if os.path.isfile(i):
            anno_folder_ini = i.replace('clips','annotations').replace('.mp4', '')
            path_clips = os.path.join(i.split('\\')[0], i.split('\\')[1]).replace(root, destination)
            path_anno = path_clips.replace('clips', 'annotations')

            if not os.path.exists(path_clips):
                os.makedirs(path_clips)
            if not os.path.exists(path_anno):
                os.makedirs(path_anno)
            
            shutil.copy(i, path_clips)
            shutil.copy(anno_folder_ini, path_anno)