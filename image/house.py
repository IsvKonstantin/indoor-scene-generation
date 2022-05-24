import os

import cv2
import numpy as np

from room import RoomInstance
from variables import *


class HouseInstance:
    def __init__(self, house_json, name, logger, img_size=256):
        self.img_size = img_size
        self.min_furniture_count = 2
        self.min_side_length = 2.5
        self.max_side_length = 8.0
        self.house_json = house_json
        self.images = dict()
        self.logger = logger
        self.name = name

    def process_rooms(self):
        self.logger.info(self.name + ": processing rooms")
        self.filter_rooms()

        for room in self.house_json:
            self.filter_furniture(room)

            if self.check_room(room):
                room_instance = RoomInstance(self.house_json[room], self.logger)
                room_instance.draw_room()
                room_instance.draw_label_map()
                room_instance.draw_plan()
                room_instance.shift_images()
                self.images[room] = {
                    "image": room_instance.img,
                    "label_map": room_instance.label_map
                }
                room_instance.check_correctness()

    def save_images(self, path_images, path_label_maps, mode=None):
        if len(self.images) < 1:
            self.logger.warning(self.name + ": no suitable rooms found")
            return

        for room_ in self.images:
            room = self.images[room_]

            room_image = room['image']

            if mode == 'Debug':
                label_image = cv2.cvtColor(room['label_map'], cv2.COLOR_GRAY2RGB)
                label_image *= 255 // label_image.max()
                border = np.full((256, 32, 3), [255, 255, 255], np.uint8)
                result = np.concatenate((room_image, border), axis=1)
                result = np.concatenate((result, label_image), axis=1)
                new = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path_images + '/' + self.name + "_" + room_ + '.png', new)
            else:
                new_image = cv2.cvtColor(room_image, cv2.COLOR_RGB2BGR)
                new_label_image = room['label_map']

                path_images = os.path.join(path_images, self.name + "_" + room_ + '.png')
                path_label_maps = os.path.join(path_label_maps, self.name + "_" + room_ + '.png')
                cv2.imwrite(path_images, new_image)
                cv2.imwrite(path_label_maps, new_label_image)

    def filter_rooms(self):
        def room_lambda(x):
            room = x[0].split('-')[0]
            if room not in SUITABLE_ROOMS:
                msg = x[0] + ": ignored" + ' --- ' + 'Reason: type is inappropriate'
                self.logger.warning(msg)
                return False
            return True

        filtered = dict(filter(room_lambda, self.house_json.items()))
        self.house_json = filtered

    def filter_furniture(self, room):
        def furniture_lambda(x):
            if (x['category'] is None) or (x['category'] in FILTER_CATEGORIES):
                msg = room + ": removed " + x['model_id'] + ' --- ' + 'Reason: category is null / inappropriate'
                self.logger.warning(msg)
                return False
            return True

        filtered = list(filter(furniture_lambda, self.house_json[room]['furniture']))
        self.house_json[room]['furniture'] = filtered

    def check_room(self, room):
        if len(self.house_json[room]['furniture']) < self.min_furniture_count:
            msg = room + ": ignored" + ' --- ' + 'Reason: not enough furniture'
            self.logger.warning(msg)
            return False

        if (not self.house_json[room]['door']) and (not self.house_json[room]['hole']):
            msg = room + ": ignored" + ' --- ' + 'Reason: no doors / holes'
            self.logger.warning(msg)
            return False

        floor = self.house_json[room]['floor']
        x_length = abs((min(np.array(floor).T[0]) - max(np.array(floor).T[0])))
        y_length = abs((min(np.array(floor).T[1]) - max(np.array(floor).T[1])))

        if (x_length < self.min_side_length) or (x_length > self.max_side_length):
            msg = room + ": ignored" + ' --- ' + 'Reason: inappropriate width / height'
            self.logger.warning(msg)
            return False

        if (y_length < self.min_side_length) or (y_length > self.max_side_length):
            msg = room + ": ignored" + ' --- ' + 'Reason: inappropriate width / height'
            self.logger.warning(msg)
            return False

        return True

