import json
import logging
import os
from tqdm import tqdm
from pywavefront import Wavefront


def verts_to_bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def bbox_size(bbox):
    # return tuple with sizes (dx, dy, dz)
    return ((bbox[1] - bbox[0]) * 100, (bbox[3] - bbox[2]) * 100, (bbox[5] - bbox[4]) * 100)


if __name__ == '__main__':
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

