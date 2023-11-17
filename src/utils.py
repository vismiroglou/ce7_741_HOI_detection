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
            return None   
    else:
        return None

def visualize_annotations(img_path, anno_path):
    import cv2
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    frame = cv2.imread(img_path)
    plt.imshow(frame)
    try:
        annos = pd.read_csv(anno_path, sep=' ', header=None)
        for row in annos.iterrows():
            x1, y1, x2, y2 = int(row[1][2]), int(row[1][3]), int(row[1][4]), int(row[1][5])
            label = row[1][0]
            plt.text(x1, y1, label, fontsize = 7, c='white')
            plt.gca().add_patch(Rectangle((x1,y1),(x2-x1),(y2-y1),
                            edgecolor='red',
                            facecolor='none',
                            lw=1))
    except:
        print('No objects in the scene')
    plt.show()

def visualize_metrics(classifier, X_test, y_test, params, i):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt

    cm = confusion_matrix(y_test, classifier.predict(X_test))
    acc = accuracy_score(y_test, classifier.predict(X_test))
    prec = precision_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    recall = recall_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    f1 = f1_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    print('Accuracy:', acc, '\nPrecision:', prec,'\nRecall:', recall,'\nF1 Score:', f1)
    
    disp = ConfusionMatrixDisplay(cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()
    
def visualize_metrics_plots(classifier, X_test, y_test, params, i):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
    from matplotlib import pyplot as plt

    cm = confusion_matrix(y_test, classifier.predict(X_test))
    acc = accuracy_score(y_test, classifier.predict(X_test))
    prec = precision_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    recall = recall_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    f1 = f1_score(y_test, classifier.predict(X_test), average='macro',zero_division=0.0)
    report = classification_report(y_test,classifier.predict(X_test),zero_division=0.0)
    print('Accuracy:', acc, '\nPrecision:', prec,'\nRecall:', recall,'\nF1 Score:', f1,'\nClassification Report:\n',report)
    
    disp = ConfusionMatrixDisplay(cm, display_labels=classifier.classes_)
    plt.figure()
    disp.plot()
    plt.title(params)
    plt.tight_layout()
    plt.savefig(f"output/cm_{i}.jpg", format="jpg")
    plt.close()
    

def create_output_video(clf, clip:str = 'data/data_inter/clips/20200519/clip_33_1450.mp4', output_dir:str = 'products/', fps:int = 10):
    import cv2
    from sklearn.preprocessing import LabelEncoder
    from src.detect_interaction import detect_interaction
    from glob import glob

    le = LabelEncoder()
    le.fit(['human-ride-bicycle', 'human-walk-bicycle', 'human-hold-bicycle', 'human-ride-motorcycle', 'human-walk-motorcycle'])

    frames_list = []
    size = (384, 288)

    #Define the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    anno_dir= clip.replace('clips', 'annotations').replace('.mp4', '')
    anno_files = sorted(glob(os.path.join(anno_dir, '*.txt')))

    for anno_path in anno_files:
        frame_path = anno_path.replace('annotations_', 'frame_').replace('annotations', 'clips').replace('.txt', '.jpg')
        frame = cv2.imread(frame_path) #frame is an actual img whereas frame_path is the path that points to the img in your local machine
        coords, predictions = detect_interaction(img_path=frame_path, anno_path=anno_path, classifier=clf)
        if predictions is not None:
            for coord, prediction in zip(coords, predictions):
                prediction = str(le.inverse_transform(prediction))
                cv2.rectangle(frame,(coord[0], coord[1]), (coord[2], coord[3]), (255,0,0),1)
                cv2.putText(frame,prediction,(coord[0],coord[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
                
        frames_list.append(frame)

    out = cv2.VideoWriter(os.path.join(output_dir,'project.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frames_list)):
        out.write(frames_list[i])
    out.release()

