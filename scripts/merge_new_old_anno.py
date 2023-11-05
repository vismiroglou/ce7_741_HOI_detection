import pandas as pd
import os

def merge_interactions(interactions:pd.DataFrame, root):

    for i in interactions.index:
        anno_folder = os.path.join(root, 'annotations', str(interactions.loc[i,'folder_name']), str(interactions.loc[i, 'clip_name']))
        verb = interactions.loc[i, 'action']
        
        for frame in range(interactions.loc[i,'frame_start'], interactions.loc[i,'frame_end'] + 1):
            print(frame)
            anno_file = os.path.join(anno_folder, 'annotations_%s.txt'%f'{frame:04}')
            with open(anno_file, 'r') as f:
                old_anno = pd.read_csv(f, header=None, sep=' ')
            try:
                human_idx = old_anno.index[old_anno[0] == interactions.loc[i,'human_id']].item()
                obj_idx = old_anno.index[old_anno[0]==interactions.loc[i,'object_id']].item()
                #Create new bounding boxes
                old_anno.loc[human_idx,2] = min(old_anno.loc[human_idx,2], old_anno.loc[obj_idx,2])
                old_anno.loc[human_idx,3] = min(old_anno.loc[human_idx,3], old_anno.loc[obj_idx,3])
                old_anno.loc[human_idx,4] = max(old_anno.loc[human_idx,4], old_anno.loc[obj_idx,4])
                old_anno.loc[human_idx,5] = max(old_anno.loc[human_idx,5], old_anno.loc[obj_idx,5])
                #Create new class name
                obj = old_anno.loc[obj_idx, 1]
                new_class = 'human-'+verb+'-'+obj
                old_anno.loc[human_idx,1] = new_class
                old_anno = old_anno.drop(obj_idx)
                
            except:
                print('Could not locate human or object in annotations', anno_file, interactions.loc[i,'human_id'], interactions.loc[i,'object_id'])
            old_anno.to_csv(anno_file, index=False, header=None, sep=' ')

if __name__ == '__main__':
    interactions = pd.read_csv(r'..\data\mini_test_folder\annotations_hoi_video_741.csv', sep=',')
    merge_interactions(interactions, r'..\data\mini_test_folder')
            