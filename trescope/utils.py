import logging

import numpy as np
import cv2
import json
import os
from tqdm import tqdm

colors = {
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'red': [255, 0, 0],
    'lime': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'silver': [192, 192, 192],
    'gray': [128, 128, 128],
    'maroon': [128, 0, 0],
    'olive': [128, 128, 0],
    'green': [0, 128, 0],
    'purple': [128, 0, 128],
    'teal': [0, 128, 128],
    'navy': [0, 0, 128]
}

furniture_colors = {
    'Cabinet/Shelf/Desk': 'lime',
    'Bed': 'purple',
    'Chair': 'blue',
    'Table': 'magenta',
    'Sofa': 'yellow',
    'Pier/Stool': 'cyan',
    'Lighting': 'red',
    'Others': 'green'
}


# noinspection PyUnresolvedReferences
def texture_color_mask(input_filename, output_filename, color):
    img = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)
    img = np.array(img, dtype=np.uint8)
    new_color = np.array(color + [255], dtype=np.uint8)
    img = np.apply_along_axis(lambda x: x if x[3] == 0 else new_color, 2, img)
    cv2.imwrite(output_filename, cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def reload_textures(model_dir_path='/home/flexatroid/PycharmProjects/data/res/3D-FRONT-samples/3D-FUTURE-model'):
    directories = [x for x in os.walk(model_dir_path)][1:]
    for dir_info in tqdm(directories):
        dir_path = dir_info[0]
        texture_path = os.path.join(dir_path, 'texture.png')
        texture_old_path = os.path.join(dir_path, 'texture_old.png')

        if os.path.isfile(texture_old_path):
            os.remove(texture_path)
            os.rename(texture_old_path, texture_path)


def modify_textures(model_dir_path='/home/flexatroid/PycharmProjects/data/res/3D-FRONT-samples/3D-FUTURE-model',
                    model_info_path='info/model_info.json'):
    with open(model_info_path) as json_file:
        data = json.load(json_file)

    new_data = list()
    for description in tqdm(data):
        model_id = description['model_id']
        if model_id == '6934dea0-1d66-49c4-82c6-4d54d41f9707':
            continue

        try:
            super_category = description['super-category']
            current_dir_path = os.path.join(model_dir_path, model_id)

            if os.path.isdir(current_dir_path):
                texture_path = os.path.join(current_dir_path, 'texture.png')
                texture_old_path = os.path.join(current_dir_path, 'texture_old.png')

                os.rename(texture_path, texture_old_path)
                description['color'] = furniture_colors[super_category]
                new_data.append(description)
                texture_color_mask(texture_old_path, texture_path, colors[description['color']])
        except Exception as e:
            logging.error("Error: " + model_id)

    with open('new_model_info.txt', 'w') as outfile:
        json.dump(new_data, outfile, indent=2)

    # with open(os.path.join(model_dir_path, 'filtered_' + model_info_path), 'w') as outfile:
    #     json.dump(new_data, outfile, indent=2)


# modify_textures('/home/flexatroid/PycharmProjects/data/res/3D-FRONT-samples/3D-FUTURE-model', 'model_info.json')
# reload_textures('/home/flexatroid/PycharmProjects/data/res/3D-FRONT-samples/3D-FUTURE-model')
