import os

from utils import *
from variables import *
from Generator.layout import LayoutInstance


class RawRoomInstance:
    def __init__(self, scene_file):
        self.scene_file = scene_file
        self.scene_content = None
        self.floor_info = None
        self.model_pool = None
        self.model_dict = dict()
        self.initialize()

    def initialize(self):
        self.scene_content = read_scene_json(self.scene_file)
        self.floor_info = get_floor_info(self.scene_file)
        self.model_pool = json.load(open('config/my_models_info.json', 'r'))

        furniture_dict = self.scene_content.dict_instance_for_furniture
        for ikey in furniture_dict.keys():
            ivalue = furniture_dict[ikey]
            uid = ikey
            jid = ivalue.jid
            if jid not in self.model_pool.keys():
                continue

            model_info = self.model_pool[jid]
            if not model_info:
                print('==> Error:', jid)
                continue

            self.model_dict[uid] = {
                'box': model_info['boundingBox'],
                'jid': jid,
                'super-category': model_info['super-category'],
                'style': model_info['style']
            }

    def parse(self, output_path):
        room_dict = self.scene_content.dict_room

        for iname in room_dict.keys():

            room = room_dict[iname]
            if room.type not in SUPPORTED_ROOMS:
                continue

            if iname not in self.floor_info.keys():
                continue

            floor = self.floor_info[iname]['floor']
            layout_info = dict()
            layout_info['room_floor'] = floor
            layout_info['seed'] = []
            layout_info['furniture'] = []

            furnitures = room.children_for_furniture
            for ifurniture in furnitures:
                uid = ifurniture['id']
                if uid not in self.model_dict.keys():
                    continue

                model = self.model_dict[uid]

                if model['super-category'] == 'Lighting':
                    layout_info['furniture'].append({
                        'jid': model['jid'],
                        'size': {
                            "xLen": 45.0,
                            "yLen": 45.0,
                            "zLen": 45.0
                        },
                        'scale': np.array([1.0, 1.0, 1.0]),
                        'pos': ifurniture['pos'],
                        'rot': ifurniture['rot']
                    })
                else:
                    layout_info['furniture'].append({
                        'jid': model['jid'],
                        'size': model['box'],
                        'scale': ifurniture['scale'],
                        'pos': ifurniture['pos'],
                        'rot': ifurniture['rot']
                    })

            layout = LayoutInstance(layout_info)

            furniture = []
            for model in layout.models:
                furniture.append({
                    'model_id': model.jid,
                    'bbox': model.get_bounding_box_scaled().tolist(),
                    'super-category': self.model_pool[model.jid]['super-category'],
                    'category': self.model_pool[model.jid]['category'],
                    "style": self.model_pool[model.jid]['style'],
                })
            self.floor_info[iname]['furniture'] = furniture

        path = os.paths.join(output_path, self.scene_content.get_json()['uid'] + '-parsed.json')
        with open(path, 'w') as outfile:
            json.dump(self.floor_info, outfile, indent=None)

