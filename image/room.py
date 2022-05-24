import cv2
import numpy as np
from variables import *

class RoomInstance:
    def __init__(self,  room_json, logger, img_size=256, scale=1.0):
        self.img_size = img_size
        self.img = None
        self.label_map = None
        self.room_json = room_json
        self.side = None
        self.xdiff = None
        self.ydiff = None
        self.scale = scale
        self.drawn_furniture = None
        self.drawn_connections = None
        self.logger = logger
        self.initialize()

    def initialize(self):
        self.xrange, self.yrange = self.room_ranges()
        self.side = max(abs(self.xrange[0] - self.xrange[1]), abs(self.yrange[0] - self.yrange[1]))
        self.xdiff = 0 - self.xrange[0]
        self.ydiff = 0 - self.yrange[0]
        self.scale = self.side / 8.0
        self.img = np.zeros((256, 256, 3), np.uint8)
        self.label_map = np.zeros((256, 256), np.uint8)
        self.plan = np.zeros((256, 256, 3), np.uint8)
        self.drawn_furniture = {}
        self.drawn_connections = {
            'floor': COLORS['silver'],
            'void': COLORS['black']
        }

    def room_ranges(self):
        floor = self.room_json['floor']
        xrange = (min(np.array(floor).T[0]), max(np.array(floor).T[0]))
        yrange = (min(np.array(floor).T[1]), max(np.array(floor).T[1]))

        if self.room_json['door']:
            for door in self.room_json['door']:
                door_ = np.reshape(np.array(door), (-1, 2))
                xrange = (min(xrange[0], min(door_.T[0])), max(xrange[1], max(door_.T[0])))
                yrange = (min(yrange[0], min(door_.T[1])), max(yrange[1], max(door_.T[1])))

        if self.room_json['hole']:
            for hole in self.room_json['hole']:
                hole_ = np.reshape(np.array(hole), (-1, 2))
                xrange = (min(xrange[0], min(hole_.T[0])), max(xrange[1], max(hole_.T[0])))
                yrange = (min(yrange[0], min(hole_.T[1])), max(yrange[1], max(hole_.T[1])))

        if self.room_json['window']:
            for window in self.room_json['window']:
                window_ = np.reshape(np.array(window), (-1, 2))
                xrange = (min(xrange[0], min(window_.T[0])), max(xrange[1], max(window_.T[0])))
                yrange = (min(yrange[0], min(window_.T[1])), max(yrange[1], max(window_.T[1])))

        return xrange, yrange

    def _rescale_x(self, x):
        new_x = round(((x + self.xdiff) * self.img_size / self.side) * self.scale)
        return min(new_x, self.img_size - 1)

    def _rescale_y(self, y):
        new_y = round(((y + self.ydiff) * self.img_size / self.side) * self.scale)
        return min(new_y, self.img_size - 1)

    def rescale(self, points):
        new_x = []
        new_y = []
        for x in points[0::2]:
            new_x.append(self._rescale_x(x))
        for y in points[1::2]:
            new_y.append(self._rescale_y(y))

        rescaled = []
        for i in range(len(new_x)):
            rescaled.append([new_x[i], new_y[i]])
        return np.array(rescaled)

    def shift_images(self):
        x0 = self._rescale_x(self.xrange[0])
        x1 = self._rescale_x(self.xrange[1])

        y0 = self._rescale_y(self.yrange[0])
        y1 = self._rescale_y(self.yrange[1])

        shift_x = self.img_size // 2 - ((x0 + x1) // 2)
        shift_y = self.img_size // 2 - ((y0 + y1) // 2)

        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_img = cv2.warpAffine(self.img, translation_matrix, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        self.img = shifted_img
        shifted_img = cv2.warpAffine(self.label_map, translation_matrix, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        self.label_map = shifted_img
        shifted_img = cv2.warpAffine(self.plan, translation_matrix, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        self.plan = shifted_img

    def fix_overlapping_chairs(self):
        updates = dict()
        chairs = dict()

        for i1, furniture1 in enumerate(self.room_json['furniture']):
            if furniture1['super-category'] == 'Chair':
                chairs[i1] = furniture1['bbox']

        for i in range(len(chairs) - 1):
            for j in range(i + 1, len(chairs)):
                indexes = list(chairs.keys())
                if indexes[i] in updates:
                    continue
                if indexes[j] in updates:
                    continue

                c1 = np.array(chairs[indexes[i]])
                c2 = np.array(chairs[indexes[j]])
                result = self.fix(c1, c2)
                if result[0]:
                    updates[indexes[i]] = result[1].tolist()
                    updates[indexes[j]] = result[2].tolist()

        for index in updates:
            self.room_json['furniture'][index]['bbox'] = updates[index]

    def fix(self, first, second):
        threshold = 0.15

        x1range = (first.T[0].max() - first.T[0].min())
        y1range = (first.T[1].max() - first.T[1].min())

        x2range = (second.T[0].max() - second.T[0].min())
        y2range = (second.T[1].max() - second.T[1].min())

        x1mid = (first.T[0].max() + first.T[0].min()) / 2
        y1mid = (first.T[1].max() + first.T[1].min()) / 2

        x2mid = (second.T[0].max() + second.T[0].min()) / 2
        y2mid = (second.T[1].max() + second.T[1].min()) / 2

        x = abs(x1mid - x2mid)
        y = abs(y1mid - y2mid)

        xd = x - x1range - x2range
        if (x - x1range / 2 - x2range / 2 < threshold) and (y < max(y1range / 2, y2range / 2)):
            delta = (threshold - (x - x1range / 2 - x2range / 2)) / 2
            if x1mid < x2mid:
                first.T[0] -= delta
                second.T[0] += delta
            else:
                first.T[0] += delta
                second.T[0] -= delta
            return [True, first, second]

        if (y - y1range / 2 - y2range / 2 < threshold) and (x < max(x1range / 2, x2range / 2)):
            delta = (threshold - (y - y1range / 2 - y2range / 2)) / 2
            if y1mid < y2mid:
                first.T[1] -= delta
                second.T[1] += delta
            else:
                first.T[1] += delta
                second.T[1] -= delta
            return [True, first, second]

        return [False, first, second]

    def draw_label_map(self):
        # draw door label
        for connection in self.room_json['door']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.label_map, rescaled, CONNECTION_CODES['door'])

        # draw hole label
        for connection in self.room_json['hole']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.label_map, rescaled, CONNECTION_CODES['hole'])

        # draw window label
        for connection in self.room_json['window']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.label_map, rescaled, CONNECTION_CODES['window'])

        # draw floor label
        rescaled = self.rescale(np.array(self.room_json['floor']).flatten())
        cv2.fillConvexPoly(self.label_map, rescaled, CONNECTION_CODES['floor'])

    def draw_plan(self):
        # draw door label
        for connection in self.room_json['door']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.plan, rescaled, COLORS[CONNECTION_CODES['door']])

        # draw hole label
        for connection in self.room_json['hole']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.plan, rescaled, COLORS[CONNECTION_CODES['hole']])

        # draw window label
        for connection in self.room_json['window']:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.plan, rescaled, COLORS[CONNECTION_CODES['window']])

        # draw floor label
        rescaled = self.rescale(np.array(self.room_json['floor']).flatten())
        cv2.fillConvexPoly(self.plan, rescaled, COLORS[CONNECTION_CODES['floor']])

    def draw_room(self):
        self.draw_connection('door')
        self.draw_connection('hole')
        self.draw_connection('window')
        self.draw_floor()

        self.draw_furniture('Bed')
        self.draw_furniture('Table')
        self.draw_furniture('Sofa')
        self.draw_furniture('Cabinet/Shelf/Desk')

        self.fix_overlapping_chairs()
        self.draw_furniture('Chair')

        self.draw_furniture('Lighting')

    def draw_floor(self):
        color = CONNECTION_COLORS['floor']
        color_rgb = COLORS[color]
        rescaled = self.rescale(np.array(self.room_json['floor']).flatten())
        cv2.fillConvexPoly(self.img, rescaled, color_rgb)

    def draw_connection(self, category):
        color = CONNECTION_COLORS[category]
        color_rgb = COLORS[color]
        for connection in self.room_json[category]:
            rescaled = self.rescale(connection)
            cv2.fillConvexPoly(self.img, rescaled, color_rgb)
            self.drawn_connections[category] = color_rgb

    def draw_furniture(self, category):
        color = FURNITURE_COLORS[category]
        color_rgb = COLORS[color]
        for furniture in self.room_json['furniture']:
            if furniture['super-category'] != category:
                continue
            bbox = np.array(furniture['bbox']).flatten()
            rescaled = self.rescale(bbox)
            cv2.fillConvexPoly(self.img, rescaled, color_rgb)
            self.drawn_furniture[category] = color_rgb

    def check_correctness(self):
        expected = set(tuple(row) for row in list(self.drawn_connections.values()) + list(self.drawn_furniture.values()))
        actual = set(tuple(row) for row in np.unique(self.img.reshape(-1, self.img.shape[2]), axis=0).tolist())
        msg = self.room_json['sceneid'] + ": " + self.room_json['instanceid'] + ": wrong image"
        assert actual == expected, msg

        expected = set(value for value in np.unique(self.label_map.flatten()).tolist())
        actual = set(CONNECTION_CODES[row] for row in list(self.drawn_connections.keys()))
        msg = self.room_json['sceneid'] + ": " + self.room_json['instanceid'] + ": wrong label"
        assert actual == expected, msg