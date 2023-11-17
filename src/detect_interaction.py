from src.utils import pair_creation
import pandas as pd
import numpy as np
import cv2

def detect_interaction(img_path:str, anno_path:str, classifier, pad_h=77, pad_w=62):
    pairs = pair_creation(anno_path)
    if pairs is not None:
        coords = []
        predictions = []
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
            x2_m = max(x2_o, x2_h)
            y2_m = max(y2_o, y2_h)

            crop = cv2.imread(img_path)[y1_m:y2_m, x1_m:x2_m]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = crop.astype(float)/255

            crop = np.pad(crop ,pad_width=(((pad_h - crop.shape[0])//2, (pad_h - crop.shape[0] + 1)//2),
                                      ((pad_w - crop.shape[1])//2, (pad_w - crop.shape[1] + 1)//2)),
                                        mode="constant", constant_values=0.0)

            crop = np.ravel(crop).reshape(1, -1)
            prediction = classifier.predict(crop)
            coords.append((x1_m, y1_m, x2_m, y2_m))
            predictions.append(prediction)
    else:
        return None, None
    return coords, predictions

