'''
The script expects the frame annotation files to be of the following format:

ID class x1 y1 x2 y2 oc inter inter_ID

ID: The unique ID of the object
class: class of the object as string type. Currently existing values are human, bicycle, motorcycle, vehicle
x1, y1: top left corner of the bounding box
x2, y2: bottom right corner of the bounding box
oc: occlusion
inter: interaction as string type. Currentlyy existing values are ride, walk 
inter_ID: The ID of the interacting object. e.g., if a human rides a bicycle then the bicycle id will exist at the end of the human line
		and the human id will exist at the end of the bicycle line. That way we can maybe support multiple interactions per scene. 
		Can be ignored if not needed.

inter and inter_ID will only exist if the file has been interaction-annotated at least once before. 

example:

00022732 human 118 97 124 116 1  
00022735 human 194 100 200 119 1 walk 00022736
00022736 bicycle 197 108 203 119 1 walk 00022735
00022737 human 126 102 132 119 1  


It also expects a SINGLE interaction-annotation file, sharing the SAME NAME as the folder it lies in (clip_X_XXXX)
with the following format:

ID1 ID2 inter sFrame eFrame

ID1: The unique ID of one of the interacting objects. Conventionally the human
ID2: The unique ID of the other interacting object.
inter: Interaction as string type
sFrame: Starting frame of the interaction. Check existing annotations in case an interaction starts when the objects come into the scene.
	Make sure their 'appearence' aligns with the existing annotations.
eFrame: End frame of the interaction. Check existing annotations in case an interaction stops when the objects leave the scene.
	Make sure their 'disappearence' aligns with the existing annotations.
'''
import os
import pandas as pd

def extract_annotations(dir):
    names_inter = ['ID1', 'ID2', 'inter', 'sFrame', 'eFrame']
    names_anno = ['ID', 'class', 'x1', 'y1', 'x2', 'y2', 'oc', 'inter', 'inter_ID']


    for folder in os.listdir(dir):
        anno_file = os.path.join(dir, folder, folder + '.txt')
        if os.path.exists(anno_file):
            annos = pd.read_csv(anno_file, sep=' ', names=names_inter, dtype=str)
            for i in annos.index:
                for frame in range(int(annos.iloc[i]['sFrame']), int(annos.iloc[i]['eFrame']) + 1):
                    frame_anno = os.path.join(dir, folder, 'annotations_' + f'{frame:04}' + '.txt')
                    if os.path.exists(frame_anno):
                        frame_anno_df = pd.read_csv(frame_anno, sep=' ', names=names_anno, dtype=str)
                        human_id = frame_anno_df.index[frame_anno_df ['ID'] == annos.iloc[i]['ID1']].to_list()
                        object_id = frame_anno_df.index[frame_anno_df ['ID'] == annos.iloc[i]['ID2']].to_list()
                        frame_anno_df.loc[human_id, 'inter'], frame_anno_df.loc[human_id, 'inter_ID'] = annos.iloc[i]['inter'], annos.iloc[i]['ID2']
                        frame_anno_df.loc[object_id, 'inter'], frame_anno_df.loc[object_id, 'inter_ID'] = annos.iloc[i]['inter'], annos.iloc[i]['ID1']
                        frame_anno_df.to_csv(frame_anno, header=None, index=None, sep=' ', mode='w')
                    else:
                        print('Annotations file', frame_anno, 'not found.')
                
        else:
            print('File', anno_file, 'was not found')
