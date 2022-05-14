import copy
import json
import math
import os
import shutil
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from trescope import Trescope
from trescope.config import FRONT3DConfig
from trescope.toolbox import simpleFileOutputs

import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess

import utils

SUPPORTED_MESH_TYPES = [
    "Hole",
    "Front",
    "Back",
    "Floor",
    "Pocket"
]

UNSUPPORTED_MESH_TYPES = [
    'SlabTop',
    'SewerPipe',
    'Flue',
    '',
    'Customized_wainscot',
    'ExtrusionCustomizedBackgroundWall',
    'Column',
    'CustomizedPlatform',
    'SlabBottom',
    'CustomizedPersonalizedModel',
    'SmartCustomizedCeiling',
    'ExtrusionCustomizedCeilingModel',
    'LightBand',
    'CustomizedCeiling',
    'Ceiling',
    'SlabSide',
]

SUPPORTED_ROOM_TYPES = [
    'LivingDiningRoom',
    'MasterBedroom',
    'SecondBedroom',
    'Bedroom',
    'BedRoom',
    'LivingRoom',
    'KidsRoom',
    'DiningRoom',
    'ElderlyRoom',
]

SUPPORTED_SUPER_CATEGORIES = [
    'Cabinet/Shelf/Desk',
    'Bed',
    'Chair',
    'Table',
    'Sofa',
    'Pier/Stool',
    'Lighting'
]

UNSUPPORTED_CATEGORIES = [
    'Bed Frame',
    'Floor Lamp',
    'Wall Lamp'
]


def visualize(output_id, model_path, file):
    (Trescope().selectOutput(output_id)
     .plotFRONT3D(file)
     .withConfig(FRONT3DConfig()
                 .view('top')
                 # .shapeLocalSource(model_path)
                 .shapeLocalSource('/home/flexatroid/Diploma/Data/3D-FUTURE-model/')
                 .hiddenMeshes(UNSUPPORTED_MESH_TYPES)
                 .renderType('color')))
    Trescope().selectOutput(output_id).flush()


def mesh_types(file_path, output_path):
    types = defaultdict(int)

    with open(file_path) as json_file:
        data = json.load(json_file)

    meshes = list(data['mesh'])

    for mesh in meshes:
        types[mesh['type']] += 1

    with open(output_path, 'w') as outfile:
        json.dump(dict(sorted(types.items(), key=lambda item: item[1])), outfile, indent=2)


def mesh_types_statistics(layouts_path='/home/flexatroid/Diploma/Data/3D-FRONT/'):
    types = defaultdict(int)
    base_dir = layouts_path
    files = os.listdir(base_dir)

    for file in tqdm(files):
        with open(os.path.join(base_dir, file)) as json_file:
            data = json.load(json_file)

        meshes = list(data['mesh'])

        for mesh in meshes:
            types[mesh['type']] += 1

    with open('mesh_types.txt', 'w') as outfile:
        json.dump(dict(sorted(types.items(), key=lambda item: item[1])), outfile, indent=2)


def rooms_statistics(layouts_path='/home/flexatroid/Diploma/Data/3D-FRONT/'):
    types = defaultdict(int)
    base_dir = layouts_path
    files = os.listdir(base_dir)

    for file in tqdm(files):
        with open(os.path.join(base_dir, file)) as json_file:
            data = json.load(json_file)

        rooms = list(data['scene']['room'])

        for room in rooms:
            types[room['type']] += 1

    with open('room_types.txt', 'w') as outfile:
        json.dump(dict(sorted(types.items(), key=lambda item: item[1])), outfile, indent=2)


def split_room(output_dir, base_dir):
    """
    ERRORS:
        1) Multiple 'Bed' in description file ('Bed' + 'Bed frame')
        2) Currently 'Others' are unsupported (Furniture types)
        3) Some rooms are irrelevant (missing floor / wrong furniture placement)
        4) Some rooms are not suitable for camera script (idk why)
        5) ...
    """
    display_textures = False

    # output_dir = "/home/flexatroid/Diploma/TestField/Output/"
    # base_dir = "/home/flexatroid/Diploma/TestField/"

    max_size = 0

    # house_path = os.path.join(base_dir, '3D-FRONT')
    house_path = '/home/flexatroid/Diploma/Data/3D-FRONT/'
    model_path = os.path.join(base_dir, '3D-FUTURE-model')
    temp_dir = os.path.join(output_dir, 'temp')
    Path(temp_dir).mkdir()

    with open('info/model_info.json') as json_file:
        model_info = json.load(json_file)

    with open('blacklists/furniture.json') as json_file:
        blacklist_furniture = json.load(json_file)

    with open('blacklists/rooms.json') as json_file:
        blacklist_rooms = json.load(json_file)

    with open('blacklists/houses.json') as json_file:
        blacklist_houses = json.load(json_file)

    model_super_categories = {}
    model_categories = {}
    for model in model_info:
        model_super_categories[model['model_id']] = model['super-category']
        model_categories[model['model_id']] = model['category']

    files = os.listdir(house_path)
    for file in tqdm(files, 'Partitioning rooms'):
        with open(os.path.join(house_path, file)) as json_file:
            data = json.load(json_file)

        if data['uid'] in blacklist_houses:
            continue

        folder_path = os.path.join(output_dir, data['uid'])
        Path(folder_path).mkdir()

        rooms = list(filter(lambda x: x['type'] in SUPPORTED_ROOM_TYPES, data['scene']['room']))
        for room in rooms:
            if room['instanceid'] in blacklist_rooms:
                continue

            if ('empty' in room) and (room['empty'] == 1):
                continue

            has_furniture = False
            for furniture in data['furniture']:
                if furniture['jid'] in model_super_categories:
                    has_furniture = True
                    break

            if not has_furniture:
                continue

            new_data = copy.deepcopy(data)
            new_data['scene']['room'] = [room]
            # new_data['material'] = list() if (not display_textures) else new_data['material']
            # new_data['materialList'] = list() if (not display_textures) else new_data['materialList']

            # 7e7f1f8e-812f-402b-b5fc-406719e9ec43
            # dbdf84ea-3a94-4c7b-b0bc-f6ca6fd2a62a_SecondBedroom-16837.json.png
            # --disable-gpu

            valid_furniture_refs = {}
            for furniture in new_data['furniture']:
                if ('valid' in furniture) and (furniture['valid']):
                    if ('jid' in furniture) and (furniture['jid'] not in blacklist_furniture):
                        if (furniture['jid'] in model_super_categories) and (furniture['jid'] in model_categories):
                            if model_super_categories[furniture['jid']] in SUPPORTED_SUPER_CATEGORIES:
                                if model_categories[furniture['jid']] not in UNSUPPORTED_CATEGORIES:
                                    valid_furniture_refs[furniture['uid']] = furniture['jid']

            # for furniture in new_data['furniture']:
            #     if furniture['jid'] in model_super_categories and 'valid' in furniture:
            #         if furniture['valid'] and model_super_categories[furniture['jid']] in SUPPORTED_SUPER_CATEGORIES:
            #             if model_categories[furniture['jid']] in UNSUPPORTED_CATEGORIES:
            #                 continue
            #             else:
            #                 valid_furniture_refs[furniture['uid']] = furniture['jid']

            new_data['furniture'] = list(filter(lambda x: x['uid'] in valid_furniture_refs, new_data['furniture']))
            new_data['furniture'] = list(map(lambda x: x if (model_super_categories[x['jid']] != 'Lighting') else x | {
                'jid': '7e7f1f8e-812f-402b-b5fc-406719e9ec43'}, new_data['furniture']))

            description = {'sceneid': new_data['uid'], 'instanceid': room['instanceid'], 'furniture': list()}
            for child in room['children']:
                if 'furniture' in child['instanceid']:
                    if child['ref'] in valid_furniture_refs:
                        description['furniture'].append(model_super_categories[valid_furniture_refs[child['ref']]])

            if description:
                room_folder_path = os.path.join(folder_path, room['instanceid'])
                room_file_path = os.path.join(temp_dir, data['uid'] + '_' + room['instanceid'] + '.json')
                room_description_path = os.path.join(room_folder_path, 'description.json')
                Path(room_folder_path).mkdir()

                with open(room_file_path, 'w') as outfile:
                    json.dump(new_data, outfile)

                subprocess.call(['python3', '3D-FRONT-ToolBox-master/run.py', room_file_path, room_description_path],
                                stdout=subprocess.DEVNULL)
                if not os.path.exists(room_description_path):
                    blacklist_rooms.append(room['instanceid'])
                    os.remove(room_file_path)
                    Path(room_folder_path).rmdir()
                    continue

                with open(room_description_path) as camera_data_file_path:
                    camera_data = json.load(camera_data_file_path)

                max_size = max(max_size, camera_data['width'], camera_data['height'])
                description.update(camera_data)

                with open(room_description_path, 'w') as outfile:
                    json.dump(description, outfile, indent=2)

    with open('blacklists/rooms.json', 'w') as outfile:
        json.dump(blacklist_rooms, outfile, indent=2)

    # h = max_size * math.sqrt(3) * 0.5
    h = 15
    print(max_size)
    print(h)
    for file in tqdm(os.listdir(temp_dir), 'Rendering files   '):
        dir_ = file.split('_')[0]
        room_ = file.split('_')[1].split('.')[0]
        description_ = os.path.join(output_dir, dir_, room_, 'description.json')

        subprocess.call(['python3', 'render_test.py', file, os.path.join(temp_dir, file), output_dir, description_, str(h)])
        time.sleep(1)

    # for file in tqdm(os.listdir(trescope_dir), 'Moving files      '):
    #     dir_ = file.split('_')[0]
    #     room_ = file.split('_')[1].split('.')[0]
    #     shutil.move(os.path.join(trescope_dir, file), os.path.join(output_dir, dir_, room_, 'render.png'))

    # output_ids = os.listdir(temp_dir)
    # Trescope().initialize(True, simpleFileOutputs(output_dir, output_ids, 256, 256))
    #
    # for file in tqdm(os.listdir(temp_dir), 'Rendering files   '):
    #     visualize(file, model_path, os.path.join(temp_dir, file))

    # for file in tqdm(os.listdir(trescope_dir), 'Moving files      '):
    #     dir_ = file.split('_')[0]
    #     room_ = file.split('_')[1].split('.')[0]
    #     shutil.move(os.path.join(trescope_dir, file), os.path.join(output_dir, dir_, room_, 'render.png'))

    # shutil.rmtree(temp_dir, ignore_errors=True)
    # shutil.rmtree(trescope_dir, ignore_errors=True)


def do_stuff(output_dir, ids, file, model_path, temp_dir):
    Trescope().initialize(True, simpleFileOutputs(output_dir, [ids], 256, 256))
    visualize(file, model_path, os.path.join(temp_dir, file))


def tretest():
    output_dir = "/home/flexatroid/Diploma/TestField/Output/"
    base_dir = "/home/flexatroid/Diploma/TestField/"
    # trescope_dir = os.path.join(output_dir, 'trescope-plot')
    # model_path = os.path.join(base_dir, '3D-FUTURE-model')
    # temp_dir = os.path.join(output_dir, 'temp')
    #
    # output_ids = os.listdir(temp_dir)
    # for file in tqdm(os.listdir(temp_dir), 'Rendering files   '):
    #     subprocess.call(['python3', 'render_test.py', file, os.path.join(temp_dir, file), output_dir])
    #
    # for file in tqdm(os.listdir(trescope_dir), 'Moving files      '):
    #     dir_ = file.split('_')[0]
    #     room_ = file.split('_')[1].split('.')[0]
    #     shutil.move(os.path.join(trescope_dir, file), os.path.join(output_dir, dir_, room_, 'render.png'))


def split_room_test(display_textures=False):
    with open('preview/preview-4.json') as json_file:
        data = json.load(json_file)

    children = data['scene']['room'][4]['children']
    refs_set = set()

    for child in children:
        refs_set.add(child['ref'])

    if not display_textures:
        data['material'] = list()

    data['furniture'] = list(filter(lambda x: x['uid'] in refs_set, data['furniture']))
    data['mesh'] = list(filter(lambda x: x['uid'] in refs_set, data['mesh']))
    data['scene']['room'] = [(data['scene']['room'][4])]

    with open('preview/preview-filtered-4.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


def remove_furniture():
    pictures_dir = "/home/flexatroid/Diploma/TestField/batch_2c/"
    jsons_dir = "/home/flexatroid/Diploma/TestField/Output/temp/"
    temp_dir = '/home/flexatroid/Diploma/TestField/temp/'

    for picture in tqdm(os.listdir(pictures_dir)):
        filename = picture[:-4]
        json_path = os.path.join(jsons_dir, filename)

        with open(json_path) as json_file:
            data = json.load(json_file)

        data["furniture"] = []

        with open(os.path.join(temp_dir, filename), 'w') as outfile:
            json.dump(data, outfile)


def render_floors():
    output_dir = "/home/flexatroid/Diploma/TestField/floors/"
    temp_dir = '/home/flexatroid/Diploma/TestField/temp/'
    cnt = 20
    process = subprocess.Popen("trescope", stdout=subprocess.DEVNULL)
    time.sleep(10)

    rendered_images = os.listdir('/home/flexatroid/Diploma/TestField/floors/trescope-plot/')

    for file in tqdm(os.listdir(temp_dir), 'Rendering images  '):
        dir_ = file.split('_')[0]
        room_ = file.split('_')[1].split('.')[0]
        description_ = os.path.join("/home/flexatroid/Diploma/TestField/Output/", dir_, room_, 'description.json')

        if (file + '.png') in rendered_images:
            print("skipped " + file)
            continue

        with open(description_) as outfile:
            data = json.load(outfile)

        if max(data['width'], data['height']) > 8.0 or max(data['width'], data['height']) < 2.0:
            continue
        cnt += 1

        print(file)
        subprocess.call(['python3', 'render_test.py', file, os.path.join(temp_dir, file), output_dir, description_, "7.0"])
        time.sleep(1)

        if cnt % 30 == 0:
            process.kill()
            os.system("pkill trescope")
            os.system("pkill trescope")
            time.sleep(5)
            process = subprocess.Popen("trescope", stdout=subprocess.DEVNULL)
            time.sleep(5)


def render_images():
    output_dir = "/home/flexatroid/Diploma/TestField/Output/"
    temp_dir = os.path.join(output_dir, 'temp')
    cnt = 20
    process = subprocess.Popen("trescope", stdout=subprocess.DEVNULL)
    time.sleep(10)

    rendered_images = os.listdir('/home/flexatroid/Diploma/TestField/Output/trescope-plot/')

    for file in tqdm(os.listdir(temp_dir), 'Rendering images  '):
        dir_ = file.split('_')[0]
        room_ = file.split('_')[1].split('.')[0]
        description_ = os.path.join(output_dir, dir_, room_, 'description.json')

        if (file + '.png') in rendered_images:
            print("skipped " + file)
            continue

        with open(description_) as outfile:
            data = json.load(outfile)

        if max(data['width'], data['height']) > 8.0 or max(data['width'], data['height']) < 2.0:
            continue
        cnt += 1

        print(file)
        subprocess.call(['python3', 'render_test.py', file, os.path.join(temp_dir, file), output_dir, description_, "7.0"])
        time.sleep(1)

        if cnt % 30 == 0:
            process.kill()
            os.system("pkill trescope")
            os.system("pkill trescope")
            time.sleep(5)
            process = subprocess.Popen("trescope", stdout=subprocess.DEVNULL)
            time.sleep(5)


def find_largest_side():
    output_dir = "/home/flexatroid/Diploma/TestField/Output/"
    temp_dir = os.path.join(output_dir, 'temp')

    sides = list()
    cnt = 0
    avg = 0.0
    maxx = 0.0
    rrr = ""

    for file in tqdm(os.listdir(temp_dir)):
        dir_ = file.split('_')[0]
        room_ = file.split('_')[1].split('.')[0]
        description_ = os.path.join(output_dir, dir_, room_, 'description.json')

        with open(description_) as outfile:
            data = json.load(outfile)

        sides.append(max(data['width'], data['height']))
        cnt += 1
        avg += max(data['width'], data['height'])
        if max(data['width'], data['height']) > maxx:
            maxx = max(data['width'], data['height'])
            rrr = room_
    print(rrr)
    print(maxx)
    print(avg / cnt)

    # fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.hist(sides, bins=50)
    plt.show()


if __name__ == '__main__':
    # utils.modify_textures('/home/flexatroid/Diploma/Python/Testing/3D-FRONT-samples/3D-FUTURE-model')

    # split_room("/home/flexatroid/Diploma/TestField/Output/", "/home/flexatroid/Diploma/TestField/")
    # find_largest_side()

    # render_images()

    # remove_furniture()
    render_floors()

    # mesh_types('preview/preview-4.json', 'types-4.txt')
    # mesh_types_statistics()
    # split_room_test(True)
    # rooms_statistics()
