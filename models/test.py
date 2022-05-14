import json
import logging
import os

from tqdm import tqdm

from pywavefront import Wavefront
import numpy as np

def verts_to_bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def bbox_size(bbox):
    # return tuple with sizes (dx, dy, dz)
    return ((bbox[1] - bbox[0]) * 100, (bbox[3] - bbox[2]) * 100, (bbox[5] - bbox[4]) * 100)


if __name__ == '__main__':
    scene = Wavefront('/home/flexatroid/Diploma/Data/3D-FUTURE-model/4d495443-cb1b-42ba-b43e-9270f18555c0/raw_model.obj')
    arr = scene.vertices
    print(bbox_size(verts_to_bbox(arr)))
    print(np.array(arr).shape)

    models = {}
    model_dir = '/home/flexatroid/Diploma/Data/3D-FUTURE-model/'
    for file in tqdm(os.listdir(model_dir)):
        if file == 'model_info.json' or file == 'categories.py':
            continue

        try:
            model_file = os.path.join(model_dir, file, 'raw_model.obj')
            scene = Wavefront(model_file)
            bbox = bbox_size(verts_to_bbox(scene.vertices))
            models[file] = [bbox[0], bbox[2], bbox[1]]
        except Exception as e:
            logging.error("Error: " + file)

    with open('bboxes.json', 'w') as outfile:
        json.dump(models, outfile, indent=2)

 # 45%|████▍     | 7398/16565 [24:18<20:23,  7.49it/s]ERROR:root:Error: 460d03f4-c8f8-3416-91eb-7d3690ec353c
 # 66%|██████▋   | 10984/16565 [36:03<31:18,  2.97it/s]ERROR:root:Error: 17c203fc-33ef-3c53-8e15-da86e53d3fdc
 # 87%|████████▋ | 14355/16565 [48:17<08:25,  4.37it/s]ERROR:root:Error: 12b1ce0f-3d2a-3211-9e05-a537e5a43e20
 # 88%|████████▊ | 14521/16565 [48:50<07:37,  4.46it/s]ERROR:root:Error: 530c3dbc-ae01-3b03-a52d-7d9e61fe4f7f
 # 88%|████████▊ | 14608/16565 [49:05<03:24,  9.58it/s]ERROR:root:Error: 7e101ef3-7722-4af8-90d5-7c562834fabd
 # 95%|█████████▌| 15783/16565 [53:08<03:48,  3.43it/s]ERROR:root:Error: e73ff703-adb1-4d3a-993d-60f6d148bec4