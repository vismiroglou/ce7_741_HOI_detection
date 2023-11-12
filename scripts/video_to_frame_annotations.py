import pandas as pd
import os

def get_object_bboxes(annos_path, anno_file):
    # Scuffed function to add bboxes to annotation file...
    # ✅1: Find path to annotations based on folder_name and clip_name from anno_file
    # ✅2: Read the corresponding anno.txt
    # ✅3: Based on anno_file, add the human and object id as well as the class and bbox
    # ✅4: Append the info to a list and return it
    # ✅5: Convert list to DataFrame and save as csv
    
    df_annos = pd.read_csv(anno_file)
    data_list = []
    for iter, row in df_annos.iterrows():
        folder_name, clip_name, fs, fe, hid, oid, act, _ = row
        anno_path = os.path.join(annos_path, str(folder_name), clip_name)
            
        for frame_id in range(fs,fe+1):
            classes = []
            bboxes = []
            for _,_,filenames in os.walk(anno_path):
                for filename in filenames:
                    if f"annotations_{frame_id:04}.txt" in filename:
                        anno = os.path.join(anno_path,filename)
                        df = pd.read_csv(anno, sep=" ", header=None, names=["obj_id", "class", "x1","y1","x2","y2","occ"])
                        
                        for obj_id in [hid, oid]:
                            row = df[df["obj_id"] == obj_id]
                            
                            if not row.empty:
                                class_name = row['class'].values[0]
                                x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']].values[0]
                                classes.append(class_name)
                                bboxes.append((x1,y1,x2,y2))
                                
            for _ in range(len(classes)):
                if len(classes) >= 2:
                    for class_ in classes:
                        if class_ == "human":
                            continue
                        else:
                            data = {
                                "folder_name": folder_name,
                                "clip_name": clip_name,
                                "frame_id": frame_id,
                                "human_id": hid,
                                "object_id": oid,
                                "object_class": class_,
                                "action": act,
                                "hmn_x1": bboxes[0][0],
                                "hmn_y1": bboxes[0][1],
                                "hmn_x2": bboxes[0][2],
                                "hmn_y2": bboxes[0][3],
                                "obj_x1": bboxes[1][0],
                                "obj_y1": bboxes[1][1],
                                "obj_x2": bboxes[1][2],
                                "obj_y2": bboxes[1][3]
                            }
                            data_list.append(data)
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    csv_name = '../../annotations_hoi_video_741.csv'
    object_annotations_path = r'../../annos/'

    data = get_object_bboxes(object_annotations_path, csv_name)
    data.to_csv("../annotations_hoi_frame_741.csv", index=False)