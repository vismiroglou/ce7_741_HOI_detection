import os
import pandas as pd
import math

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))



class Averager(): 
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def find_center(x1, y1, x2, y2):
    xc = int((x1 + x2)/2)
    yc = int((y1 + y2)/2)
    return xc, yc

def pair_creation(anno_file_path:str):
    '''
    Returns a list of tuples of pairs.
    Returns None if no pairs were found
    '''
    if os.stat(anno_file_path).st_size !=0:
        annotations = pd.read_csv(anno_file_path, sep=' ', header=None)
        if 'bicycle' in annotations[1].values or 'motorcycle' in annotations[1].values:
            pairs=[]
            for obj in annotations[(annotations[1] == 'bicycle') | (annotations[1] == 'motorcycle')].index:
                min_distance = math.inf
                min_distance_id = 0
                xc_o, yc_o = find_center(annotations.loc[obj, 2], 
                                         annotations.loc[obj, 3], 
                                         annotations.loc[obj, 4], 
                                         annotations.loc[obj, 5])
                for idx in annotations.index.drop(obj):
                    if annotations.loc[idx, 1] == 'human':
                        xc_h, yc_h = find_center(annotations.loc[idx, 2], 
                                                 annotations.loc[idx, 3], 
                                                 annotations.loc[idx, 4], 
                                                 annotations.loc[idx, 5])
                
                        distance = math.dist([xc_o, yc_o],[xc_h, yc_h])
                        if distance < min_distance:
                            min_distance = distance
                            min_distance_id = annotations.loc[idx, 0]
                pairs.append((annotations.loc[obj, 0], min_distance_id))
            return(pairs)
        else:
            print('No interactions in the scene')
            return None   
    else:
        print('No objects in the scene')
        return None

def visualize_annotations(img_path, anno_path):
    import cv2
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    frame = cv2.imread(img_path)
    plt.imshow(frame)
    annos = pd.read_csv(anno_path, sep=' ', header=None)
    for row in annos.iterrows():
        x1, y1, x2, y2 = int(row[1][2]), int(row[1][3]), int(row[1][4]), int(row[1][5])
        label = row[1][0]
        plt.text(x1, y1, label, fontsize = 12, c='white')
        plt.gca().add_patch(Rectangle((x1,y1),(x2-x1),(y2-y1),
                        edgecolor='red',
                        facecolor='none',
                        lw=1))
    plt.show()


