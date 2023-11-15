from src.utils import pair_creation
import pandas as pd
import cv2

def detect_interaction(frame_path:str, anno_path:str, classifier):
    pairs = pair_creation(anno_path)
    if pairs is not None:
        crops=[]
        annotations = pd.read_csv(anno_path, sep=' ', header=None)
        for pair in pairs:
            x1_o = annotations[annotations[0] == pair[0]][2].item()
            y1_o = annotations[annotations[0] == pair[0]][3].item()
            x2_o = annotations[annotations[0] == pair[0]][4].item()
            y2_o = annotations[annotations[0] == pair[0]][5].item()

            x1_h = annotations[annotations[0] == pair[1]][2].item()
            y1_h = annotations[annotations[0] == pair[1]][3].item()
            x2_h = annotations[annotations[0] == pair[1]][4].item()
            y2_h = annotations[annotations[0] == pair[1]][5].item()

            x1_m = min(x1_o, x1_h)
            y1_m = min(y1_o, y1_h)
            x2_m = min(x2_o, x2_h)
            y2_m = min(y2_o, y2_h)

            crop = cv2.imread(frame_path)[y1_m:y2_m, x1_m:x2_m]
            crops.append[crop]
    else:
        return None
    return crops

