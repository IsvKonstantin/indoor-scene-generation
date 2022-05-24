import cv2
import matplotlib.pyplot as plt
import json
import os
from image.variables import *

import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from simple_3dviz import Mesh, Scene, Lines, TexturedMesh, render
from simple_3dviz.behaviours.io import SaveGif
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.renderables.textured_mesh import Material
from simple_3dviz.window import show
import tripy
import random


class ImageInstance:
    def __init__(self, img, label_map, opt):
        self.opt = opt
        self.img = img
        self.label_map = label_map
        self.initialize()

    def initialize(self):
        self.models_info = json.load(open('scene/config/my_models_info.json'))
        self.texture_info = json.load(open('scene/config/texture_info.json'))
        self.path_to_models = self.opt.path_to_models
        self.path_to_textures = self.opt.path_to_textures

        assert self.img.shape[:2] == self.label_map.shape[:2], "Images have different shapes"
        self.img_size = self.img.shape[0]
        self.m_in_pixel = (8.3 / 256)

        self.wall_color = self.opt.wall_color
        self.floor_texture = "14c80153-146d-4501-9f4e-7379ed7a60f5"
        self.door_height = 2.1
        self.wall_height = 2.7
        self.window_start = 0.8
        self.window_height = 1.7
        self.style = self.opt.style

        self.floor_mesh = None
        self.wall_meshes = None

        self.furniture_meshes = dict()
        self.furniture_rotations = dict()
        self.furniture_models = dict()
        self.furniture_clusters = dict()

        self.center = None

        self.bboxes = dict()
        self.renderables = list()

        if self.opt.path_to_config is not None:
            self.load_config()
        else:
            if self.opt.floor_texture is None:
                self.floor_texture = self.sample_texture(self.style)
            else:
                self.floor_texture = self.opt.floor_texture

    def render_image(self):
        if self.opt.show_axes:
            axes = Lines.axes((0, 0, 0), width=0.05)
            self.renderables.append(axes)

        self.renderables.append(self.floor_mesh)

        if self.opt.render_walls:
            for wall_mesh in self.wall_meshes:
                self.renderables.append(wall_mesh)

        for category in self.furniture_meshes:
            for i in self.furniture_meshes[category]:
                self.renderables.append(self.furniture_meshes[category][i])

        camera_params = self.compute_camera_params()

        if self.opt.render_mode == "auto":
            show(
                self.renderables,
                size=camera_params['size'],
                camera_position=camera_params['camera_position'],
                camera_target=camera_params['camera_target'],
                up_vector=camera_params['up_vector'],
                behaviours=[
                    CameraTrajectory(
                        Circle(
                            camera_params['circle_center'],
                            camera_params['circle_start'],
                            camera_params['up_vector']
                        ),
                        speed=camera_params['speed']
                    ),
                    LightToCamera()
                ],
                light=camera_params['light'],
                background=camera_params['background']
            )
        elif self.opt.render_mode == "default":
            show(
                self.renderables,
                size=camera_params['size'],
                camera_position=camera_params['camera_position'],
                camera_target=camera_params['camera_target'],
                up_vector=camera_params['up_vector'],
                light=camera_params['light'],
                background=camera_params['background']
            )
        elif self.opt.render_mode == "gif":
            render(
                self.renderables,
                n_frames=200,
                size=camera_params['gif_size'],
                camera_target=camera_params['camera_target'],
                camera_position=camera_params['circle_start'],
                up_vector=camera_params['up_vector'],
                behaviours=[
                    CameraTrajectory(
                        Circle(
                            camera_params['circle_center'],
                            camera_params['circle_start'],
                            camera_params['up_vector']
                        ),
                        speed=camera_params['gif_speed']
                    ),
                    LightToCamera(),
                    SaveGif('example.gif')
                ],
                background=camera_params['background']
            )

    def process_image(self):
        self.restore_floor()
        self.restore_connections()
        self.restore_walls()

        self.restore_furniture('Lighting')
        self.restore_furniture('Chair')
        self.restore_furniture('Cabinet/Shelf/Desk')
        self.restore_furniture('Sofa')
        self.restore_furniture('Table')
        self.restore_furniture('Bed')

        # remove bboxes for missing categories
        # self.bboxes = dict(filter(lambda x: len(x[1]) > 0, self.bboxes.items()))

        self.load_furniture('Bed', self.style)
        self.load_furniture('Table', self.style)
        self.load_furniture('Cabinet/Shelf/Desk', self.style)
        self.load_furniture('Sofa', self.style)
        self.load_furniture('Chair', self.style)
        self.load_furniture('Lighting', self.style)

        if self.opt.save_config:
            self.save_config()

        self.restore_rotations('Bed')
        self.restore_rotations('Cabinet/Shelf/Desk')
        self.restore_rotations('Sofa')
        self.restore_rotations('Chair')
        self.rotate_meshes()

    def load_config(self):
        config = json.load(open(self.opt.path_to_config))
        self.wall_color = config["wall_color"]
        self.floor_texture = config["floor_texture"]

        for category in config["models"]:
            self.furniture_models[category] = dict()
            for index in config["models"][category]:
                self.furniture_models[category][int(index)] = config["models"][category][index]

    def save_config(self):
        config = {
            "wall_color": self.wall_color,
            "floor_texture": self.floor_texture,
            "models": dict()
        }

        for category in self.furniture_models:
            config["models"][category] = dict()
            for i in self.furniture_models[category]:
                config["models"][category][int(i)] = self.furniture_models[category][i]

        with open('config.json', 'w') as outfile:
            json.dump(config, outfile, indent=2)

    def compute_camera_params(self):
        params = dict()
        floor_width = abs(self.floor_polygon.T[0].min() - self.floor_polygon.T[0].max())
        floor_height = abs(self.floor_polygon.T[1].min() - self.floor_polygon.T[1].max())
        radius = np.sqrt(floor_width ** 2 + floor_height ** 2) * 0.5

        params['camera_position'] = [self.center[0] + radius + 1, 8, self.center[1]]
        params['background'] = [1, 1, 1, 1]
        params['gif_size'] = [600, 600]
        params['size'] = [800, 800]
        params['up_vector'] = [0, 1, 0]
        params['camera_target'] = [self.center[0], 0, self.center[1]]
        params['circle_center'] = [self.center[0], 8, self.center[1]]
        params['circle_start'] = [self.center[0] + 1.5 * radius, 8, self.center[1]]
        params['gif_speed'] = 0.01
        params['speed'] = 0.005
        params['light'] = [self.center[0], 4, self.center[1]]
        return params

    def rotate_meshes(self):
        for category in self.furniture_meshes:
            for i in self.furniture_meshes[category]:
                mesh = self.furniture_meshes[category][i]
                mesh.rotate_y(self.furniture_rotations[category][i])

    def rerestore_chairs_rotations(self):
        for cluster in self.furniture_clusters['Chair']:
            for table_i, (_, table_rect) in enumerate(self.bboxes['Table']):
                distances = list()
                table_rect = np.array(table_rect).astype('float32') * self.m_in_pixel
                for cluster_i in cluster:
                    _, chair_rect = self.bboxes['Chair'][cluster_i]
                    chair_rect = cv2.minAreaRect(chair_rect)
                    chair_center = np.array(chair_rect[0]).astype('float32') * self.m_in_pixel
                    distances.append(cv2.pointPolygonTest(table_rect, chair_center, measureDist=True))

                distances = distances * -1
                if np.all(np.array(distances) < 0.4):
                    table_rect = self.sorted_rect(table_rect)
                    table_rect = np.row_stack((table_rect, table_rect[0]))

                    for cluster_i in cluster:
                        _, chair_rect = self.bboxes['Chair'][cluster_i]
                        chair_center = np.array(cv2.minAreaRect(chair_rect)[0]) * self.m_in_pixel
                        table_center = np.array(cv2.minAreaRect(table_rect)[0])
                        outside = cv2.pointPolygonTest(table_rect, chair_center, measureDist=False) < 0
                        side_distances = list()

                        for i in range(len(table_rect) - 1):
                            point_start = np.array(table_rect[i])
                            point_end = np.array(table_rect[i + 1])

                            n = (point_end - point_start) / np.linalg.norm(point_end - point_start, 2)
                            projected = point_start + n * np.dot(chair_center - point_start, n)

                            vector = np.array(projected - chair_center)
                            distance = np.linalg.norm(projected - chair_center, 2)
                            if not outside:
                                vector = np.array(chair_center - projected)
                            side_distances.append([abs(distance), vector, projected])

                        side_distances.sort(key=lambda x: x[0])

                        if (abs(side_distances[0][0] - side_distances[1][0]) < 0.1):
                            p1 = side_distances[0][2]
                            p2 = side_distances[1][2]

                            distance_1 = abs(cv2.pointPolygonTest(table_rect[:-1], p1, measureDist=True))
                            distance_2 = abs(cv2.pointPolygonTest(table_rect[:-1], p2, measureDist=True))
                            if distance_1 > distance_2:
                                side_distances[0], side_distances[1] = side_distances[1], side_distances[0]

                        vector = np.array(side_distances[0][1])
                        vector = vector / np.linalg.norm(vector)
                        angle = (np.arctan2(0, 1) - np.arctan2(vector[0], vector[1])) % (2 * np.pi)
                        if cluster_i in self.furniture_rotations['Chair']:
                            self.furniture_rotations['Chair'][cluster_i] = angle

                    break

    def restore_rotations(self, category):
        if category not in self.furniture_meshes:
            return

        if category == "Chair":
            self.rerestore_chairs_rotations()
            return

        floor_polygon = self.floor_polygon.tolist()
        floor_polygon.append(floor_polygon[0])

        clipped = dict()

        for i in range(len(floor_polygon) - 1):
            point_start = floor_polygon[i]
            point_end = floor_polygon[i + 1]

            x1 = point_start[0]
            z1 = point_start[1]
            x2 = point_end[0]
            z2 = point_end[1]

            dx = x2 - x1
            dz = z2 - z1

            normal = np.array([0, 0, 0])
            normal1 = np.array([-dz, dx]) / max(abs(dz), abs(dx))
            normal2 = np.array([dz, -dx]) / max(abs(dz), abs(dx))

            if (self.point_is_in_floor((x1 + x2) / 2 + normal1[0] * 0.15, (z1 + z2) / 2 + normal1[1] * 0.15)):
                normal = np.array([normal1[0], 0, normal1[1]])
            else:
                normal = np.array([normal2[0], 0, normal2[1]])

            for i, (_, rect) in enumerate(self.bboxes[category]):
                if i not in clipped:
                    clipped[i] = list()

                rect = np.array(rect).astype('float32') * self.m_in_pixel
                distances = list()
                for p in rect:
                    wall = np.row_stack((point_start, point_end)).astype('float32')
                    distances.append(abs(cv2.pointPolygonTest(wall, p, measureDist=True)))
                distances.sort()
                if distances[0] < 0.2 and distances[1] < 0.2:
                    if np.isclose(normal, np.array([0, 0, 1]), atol=0.01).all():
                        clipped[i].append("up")
                    if np.isclose(normal, np.array([0, 0, -1]), atol=0.01).all():
                        clipped[i].append("down")
                    if np.isclose(normal, np.array([-1, 0, 0]), atol=0.01).all():
                        clipped[i].append("right")
                    if np.isclose(normal, np.array([1, 0, 0]), atol=0.01).all():
                        clipped[i].append("left")

        for i in clipped.keys():
            if clipped[i]:
                if ("right" in clipped[i]) and ("down" in clipped[i]):
                    _, rect = self.bboxes[category][i]
                    rect = self.sorted_rect(rect)
                    rect_side_1 = np.linalg.norm(rect[1] - rect[0])
                    rect_side_2 = np.linalg.norm(rect[2] - rect[1])

                    if (rect_side_1 < rect_side_2) and (category == "Bed"):
                        mesh = self.furniture_meshes[category][i]
                        mesh.rotate_y(np.pi)
                    elif (rect_side_1 > rect_side_2) and (category != "Bed"):
                        mesh = self.furniture_meshes[category][i]
                        mesh.rotate_y(np.pi)
                    elif (rect_side_1 / rect_side_2 > 0.85) and (category != "Bed"):
                        mesh = self.furniture_meshes[category][i]
                        mesh.rotate_y(np.pi)
                    continue

                if ("left" in clipped[i]) and ("down" in clipped[i]):
                    mesh = self.furniture_meshes[category][i]
                    mesh.rotate_y(np.pi)
                    continue

                if ("right" in clipped[i]) and ("up" in clipped[i]):
                    # do nothing
                    continue

                if ("left" in clipped[i]) and ("up" in clipped[i]):
                    rect = self.sorted_rect(rect)
                    rect_side_1 = np.linalg.norm(rect[1] - rect[0])
                    rect_side_2 = np.linalg.norm(rect[2] - rect[1])

                    if (rect_side_1 > rect_side_2) and (category == "Bed"):
                        mesh = self.furniture_meshes[category][i]
                        mesh.rotate_y(np.pi)
                    elif (rect_side_1 < rect_side_2) and (category != "Bed"):
                        mesh = self.furniture_meshes[category][i]
                        mesh.rotate_y(np.pi)
                    #                     elif (rect_side1 / rect_side2 > 0.85) and (category != "Bed"):
                    #                         mesh = self.furniture_meshes[category][i]
                    #                         mesh.rotate_y(np.pi)
                    continue

                if ("left" in clipped[i]) and ("up" not in clipped[i]) and ("down" not in clipped[i]):
                    # small square-like objects?
                    mesh = self.furniture_meshes[category][i]
                    mesh.rotate_y(np.pi)
                    continue

                if ("right" in clipped[i]) and ("up" not in clipped[i]) and ("down" not in clipped[i]):
                    # small square-like objects?
                    # do nothing
                    continue

                if ("up" in clipped[i]) and ("left" not in clipped[i]) and ("right" not in clipped[i]):
                    # do nothing
                    continue

                if ("down" in clipped[i]) and ("left" not in clipped[i]) and ("right" not in clipped[i]):
                    mesh = self.furniture_meshes[category][i]
                    mesh.rotate_y(np.pi)
                    continue

    def restore_walls(self):
        floor_polygon = self.floor_polygon.tolist()
        floor_polygon.append(floor_polygon[0])
        self.wall_meshes = list()

        for i in range(len(floor_polygon) - 1):
            point_start = floor_polygon[i]
            point_end = floor_polygon[i + 1]

            x1 = point_start[0]
            z1 = point_start[1]
            x2 = point_end[0]
            z2 = point_end[1]

            holes = list()
            normal = np.array([0, 0, 0])

            dx = x2 - x1
            dz = z2 - z1
            normal1 = np.array([-dz, dx]) / max(abs(dz), abs(dx))
            normal2 = np.array([dz, -dx]) / max(abs(dz), abs(dx))

            if (self.point_is_in_floor((x1 + x2) / 2 + normal1[0] * 0.15, (z1 + z2) / 2 + normal1[1] * 0.15)):
                normal = np.array([normal1[0], 0, normal1[1]])
            else:
                normal = np.array([normal2[0], 0, normal2[1]])

            for door_ in self.bboxes['door']:
                door = np.array(door_[0]) * self.m_in_pixel
                door_start = door[0][0]
                door_end = door[1][0]

                if (not self.point_is_on_wall(point_start, point_end, door_start)) or \
                        (not self.point_is_on_wall(point_start, point_end, door_end)):
                    continue

                if (x1 < x2) and (door_start[0] > door_end[0]):
                    door_start, door_end = door_end, door_start

                if (x1 > x2) and (door_start[0] < door_end[0]):
                    door_start, door_end = door_end, door_start

                if (x1 == x2) and (z1 < z2) and (door_start[1] > door_end[1]):
                    door_start, door_end = door_end, door_start

                if (x1 == x2) and (z1 > z2) and (door_start[1] < door_end[1]):
                    door_start, door_end = door_end, door_start

                holes.append(("door", [door_start, door_end]))

            for window_ in self.bboxes['window']:
                window = np.array(window_[0]) * self.m_in_pixel
                window_start = window[0][0]
                window_end = window[1][0]

                if (not self.point_is_on_wall(point_start, point_end, window_start)) or \
                        (not self.point_is_on_wall(point_start, point_end, window_end)):
                    continue

                if (x1 < x2) and (window_start[0] > window_end[0]):
                    window_start, window_end = window_end, window_start

                if (x1 > x2) and (window_start[0] < window_end[0]):
                    window_start, window_end = window_end, window_start

                if (x1 == x2) and (z1 < z2) and (window_start[1] > window_end[1]):
                    window_start, window_end = window_end, window_start

                if (x1 == x2) and (z1 > z2) and (window_start[1] < window_end[1]):
                    window_start, window_end = window_end, window_start

                holes.append(("window", [window_start, window_end]))

            if (x1 < x2):
                holes.sort(key=lambda x: x[1][0][0], reverse=False)
            elif (x1 > x2):
                holes.sort(key=lambda x: x[1][0][0], reverse=True)
            elif (x1 == x2) and (z1 < z2):
                holes.sort(key=lambda x: x[1][0][1], reverse=False)
            elif (x1 == x2) and (z1 > z2):
                holes.sort(key=lambda x: x[1][0][1], reverse=True)

            current = np.array([x1, z1])
            for hole_ in holes:
                hx1, hz1 = hole_[1][0]
                hx2, hz2 = hole_[1][1]

                wall_segment = np.array([
                    [current[0], 0, current[1]],
                    [current[0], self.wall_height, current[1]],
                    [hx1, self.wall_height, hz1],
                    [hx1, 0, hz1]
                ])

                if hole_[0] == "door":
                    door_segment = np.array([
                        [hx1, self.door_height, hz1],
                        [hx1, self.wall_height, hz1],
                        [hx2, self.wall_height, hz2],
                        [hx2, self.door_height, hz2]
                    ])
                elif hole_[0] == "window":
                    window_down = np.array([
                        [hx1, 0, hz1],
                        [hx1, self.window_start, hz1],
                        [hx2, self.window_start, hz2],
                        [hx2, 0, hz2]
                    ])
                    window_up = np.array([
                        [hx1, self.window_start + self.window_height, hz1],
                        [hx1, self.wall_height, hz1],
                        [hx2, self.wall_height, hz2],
                        [hx2, self.window_start + self.window_height, hz2]
                    ])

                if x1 == x2:
                    wall_segment = wall_segment[:, [1, 2]]
                    line_lambda = lambda v: x2
                    indexes = [0, 1, 1]
                    flip = True
                else:
                    line_lambda = lambda v: ((v - x1) / (x2 - x1) * (z2 - z1) + z1)
                    wall_segment = wall_segment[:, [0, 1]]
                    indexes = [1, 1, 0]
                    flip = False

                self.wall_meshes.append(self.mesh_from_vertices(
                    wall_segment,
                    line_lambda,
                    indexes=indexes,
                    normal=normal.tolist(),
                    flip=flip,
                    texture=None,
                    color=self.wall_color
                ))

                if hole_[0] == "door":
                    self.wall_meshes.append(self.mesh_from_vertices(
                        door_segment[:, [1, 2]] if x1 == x2 else door_segment[:, [0, 1]],
                        line_lambda,
                        indexes=indexes,
                        normal=normal.tolist(),
                        flip=flip,
                        texture=None,
                        color=self.wall_color
                    ))
                elif hole_[0] == "window":
                    self.wall_meshes.append(self.mesh_from_vertices(
                        window_down[:, [1, 2]] if x1 == x2 else window_down[:, [0, 1]],
                        line_lambda,
                        indexes=indexes,
                        normal=normal.tolist(),
                        flip=flip,
                        texture=None,
                        color=self.wall_color
                    ))
                    self.wall_meshes.append(self.mesh_from_vertices(
                        window_up[:, [1, 2]] if x1 == x2 else window_up[:, [0, 1]],
                        line_lambda,
                        indexes=indexes,
                        normal=normal.tolist(),
                        flip=flip,
                        texture=None,
                        color=self.wall_color
                    ))
                current = np.array([hx2, hz2])

            wall_segment = np.array([
                [current[0], 0, current[1]],
                [current[0], self.wall_height, current[1]],
                [x2, self.wall_height, z2],
                [x2, 0, z2]
            ])

            if x1 == x2:
                line_lambda = lambda v: x2
            else:
                line_lambda = lambda v: ((v - x1) / (x2 - x1) * (z2 - z1) + z1)

            self.wall_meshes.append(self.mesh_from_vertices(
                wall_segment[:, [1, 2]] if x1 == x2 else wall_segment[:, [0, 1]],
                line_lambda,
                indexes=[0, 1, 1] if x1 == x2 else [1, 1, 0],
                normal=normal.tolist(),
                flip=True if x1 == x2 else False,
                texture=None,
                color=self.wall_color
            ))

    def restore_floor(self):
        floor_mask = cv2.inRange(self.label_map, CONNECTION_CODES['floor'], CONNECTION_CODES['floor'])
        contours, hierarchy = cv2.findContours(floor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # moments = cv2.moments(contours[0])

        floor_poly = contours[0].reshape(contours[0].shape[0], 2)
        floor_poly = floor_poly.astype(float)

        # calculating geometric center (centroid) of floor polygon
        # center_x = (moments["m10"] / moments["m00"] - floor_poly.T[0].min()) * self.m_in_pixel
        # center_z = (moments["m01"] / moments["m00"] - floor_poly.T[1].min()) * self.m_in_pixel

        # shifting floor polygon to the start of the coordinates is unessential
        # floor_poly.T[0] -= floor_poly.T[0].min()
        # floor_poly.T[1] -= floor_poly.T[1].min()

        floor_poly.T[0] *= self.m_in_pixel
        floor_poly.T[1] *= self.m_in_pixel

        center_x = 0.5 * (floor_poly.T[0].min() + floor_poly.T[0].max())
        center_z = 0.5 * (floor_poly.T[1].min() + floor_poly.T[1].max())

        self.center = (center_x, center_z)
        self.floor_polygon = floor_poly
        self.floor_mesh = self.mesh_from_vertices(
            floor_poly,
            lambda x: 0,
            indexes=[1, 0, 1],
            normal=[0, 1, 0],
            flip=False,
            texture=self.floor_texture,
            color=None
        )

    def restore_furniture(self, category):
        if category == 'Table':
            self._restore_tables()
            return

        color_hsv = cv2.cvtColor(np.uint8([[COLORS[FURNITURE_COLORS[category]]]]), cv2.COLOR_RGB2HSV)
        lower = color_hsv[0][0][0] - 10, 100, color_hsv[0][0][2] - 25
        upper = color_hsv[0][0][0] + 10, 255, color_hsv[0][0][2] + 25
        image_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.bboxes[category] = list()
        furniture_mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))
        contours, hierarchy = cv2.findContours(furniture_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = self.filter_contours(contours, hierarchy)

        for contour in contours:
            rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
            if self.fits_area(contour):
                self.bboxes[category].append((contour, rect))
                continue

            if self.contour_is_overlapped(contour):
                self.bboxes[category].append((contour, rect))
                continue
            else:
                contour_mask = np.zeros(self.label_map.shape, dtype='uint8')
                cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                masked = cv2.bitwise_and(image_hsv, image_hsv, mask=contour_mask)
                masked = cv2.inRange(masked, np.array(lower), np.array(upper))

                for kernel_dim in range(1, 10):
                    kernel = np.ones((kernel_dim, kernel_dim), 'uint8')
                    eroded = cv2.erode(masked, kernel, iterations=1)
                    sub_contours, sub_hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    sub_contours, sub_hierarchy = self.filter_contours(sub_contours, sub_hierarchy)

                    if self.all_fit(sub_contours, 0.9):
                        shrink_size = kernel_dim // 2
                        for sub_contour in sub_contours:
                            moments = cv2.moments(sub_contour)
                            center_x = int(moments["m10"] / moments["m00"])
                            center_z = int(moments["m01"] / moments["m00"])
                            for row in sub_contour:
                                row[0][0] += shrink_size * (1 if row[0][0] > center_x else -1)
                                row[0][1] += shrink_size * (1 if row[0][1] > center_z else -1)
                            sub_rect = np.int0(cv2.boxPoints(cv2.minAreaRect(sub_contour)))
                            self.bboxes[category].append((sub_contour, sub_rect))
                        break
                # other shapes are undetectable (e.g. 'L-shape' - 2 intersecting models)

    def _restore_tables(self):
        color = FURNITURE_COLORS['Table']
        color_rgb = np.array(COLORS[color])
        self.bboxes['Table'] = list()

        color_hsv = cv2.cvtColor(np.uint8([[COLORS[FURNITURE_COLORS['Table']]]]), cv2.COLOR_RGB2HSV)
        lower = color_hsv[0][0][0] - 10, 100, color_hsv[0][0][2] - 25
        upper = color_hsv[0][0][0] + 10, 255, color_hsv[0][0][2] + 25
        image_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        furniture_mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        connected = cv2.morphologyEx(furniture_mask, cv2.MORPH_CLOSE, structuring_element)

        contours, hierarchy = cv2.findContours(connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = self.filter_contours(contours, hierarchy)

        for contour in contours:
            rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
            self.bboxes['Table'].append((contour, rect))

    def restore_connections(self):
        window_mask = cv2.inRange(self.label_map, CONNECTION_CODES['window'], CONNECTION_CODES['window'])
        door_mask = cv2.inRange(self.label_map, CONNECTION_CODES['door'], CONNECTION_CODES['door'])
        floor_mask = cv2.inRange(self.label_map, CONNECTION_CODES['floor'], CONNECTION_CODES['floor'])
        self.bboxes['window'] = list()
        self.bboxes['door'] = list()

        contours_f, hierarchy_f = cv2.findContours(floor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_d, hierarchy_d = cv2.findContours(door_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_w, hierarchy_w = cv2.findContours(window_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blank = np.zeros((self.img_size, self.img_size), dtype='uint8')
        doors = cv2.bitwise_and(
            cv2.drawContours(blank.copy(), contours_d, -1, 1, 2),
            cv2.drawContours(blank.copy(), contours_f, -1, 1, 1)
        )
        contours_doors, _ = cv2.findContours(doors, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blank = np.zeros((self.img_size, self.img_size), dtype='uint8')
        windows = cv2.bitwise_and(
            cv2.drawContours(blank.copy(), contours_w, -1, 1, 2),
            cv2.drawContours(blank.copy(), contours_f, -1, 1, 1)
        )
        contours_windows, _ = cv2.findContours(windows, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_doors:
            rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
            self.bboxes['door'].append((contour, rect))

        for contour in contours_windows:
            rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
            self.bboxes['window'].append((contour, rect))

    def point_is_on_wall(self, wall_start, wall_end, point, threshold=0.05):
        a, b, p = wall_start, wall_end, point

        cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
        if (abs(cross) > threshold):
            return False

        dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
        if (dot < 0):
            return False

        squared = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
        if (dot > squared):
            return False

        return True

    def point_is_in_floor(self, x, z):
        floor_mask = cv2.inRange(self.label_map, CONNECTION_CODES['floor'], CONNECTION_CODES['floor'])
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x *= (256.0 / 8.3)
        z *= (256.0 / 8.3)

        return cv2.pointPolygonTest(contours[0], (int(x), int(z)), measureDist=False) >= 0

    def contour_is_overlapped(self, contour):
        floor_color_rgb = np.array(COLORS[CONNECTION_COLORS['floor']])

        rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
        mask = np.zeros((self.img_size, self.img_size), dtype='uint8')
        cv2.drawContours(mask, [rect], 0, 255, -1)
        inverted_mask = cv2.bitwise_not(mask)
        source = np.full((self.img_size, self.img_size, 3), 255, dtype='uint8')

        foreground = cv2.bitwise_and(self.img, self.img, mask=mask)
        background = cv2.bitwise_and(source, source, mask=inverted_mask)
        result = cv2.add(background, foreground)

        lower = np.array([0, 0, 192 - 30])
        upper = np.array([179, 60, 192 + 30])
        result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

        in_area = cv2.countNonZero(mask)
        in_area_floor = cv2.countNonZero(cv2.inRange(result_hsv, lower, upper))
        return (in_area_floor == 0) or (in_area_floor / in_area < 0.1)

    def all_fit(self, contours, threshold=0.9):
        for contour in contours:
            if not self.contour_is_overlapped(contour):
                if not self.fits_area(contour, threshold):
                    return False
        return True

    def fits_area(self, contour, threshold=0.9):
        expected = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        actual = rect[1][0] * rect[1][1]
        return (expected / actual) > threshold

    def all_rectangles(self, contours):
        for contour in contours:
            if np.squeeze(contour).shape[0] != 4:
                return False
        return True

    def filter_contours(self, contours, hierarchy):
        if hierarchy is None:
            return contours, hierarchy

        indexes = list()
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:
                indexes.append(i)
        new_hierarchy = np.array([[hierarchy[0][i] for i in indexes]])
        new_contours = tuple([contours[i] for i in indexes])
        return new_contours, new_hierarchy

    def print_bboxes(self):
        counter = 1
        fig = plt.figure(figsize=(10, 4 * (len(self.bboxes) + 1)))
        for category in self.bboxes:
            bboxes_ = self.bboxes[category]
            fig.add_subplot((len(self.bboxes) + 1), 2, counter)
            img_copy1 = self.img.copy()
            img_copy2 = self.img.copy()
            for contour, rect in bboxes_:
                cv2.drawContours(img_copy1, [contour], 0, (255, 255, 255), 2)
                cv2.drawContours(img_copy2, [rect], 0, (255, 255, 255), 2)
            plt.imshow(img_copy1)
            plt.axis('off')
            plt.title(category + ": " + str(len(bboxes_)))

            fig.add_subplot((len(self.bboxes) + 1), 2, counter + 1)
            plt.imshow(img_copy2)
            plt.axis('off')
            plt.title('bounding rects')

            counter += 2

    def sorted_rect(self, rect):
        mins = np.where(rect.T[1] == rect.T[1].min())[0]
        if len(mins) > 1:
            if rect[mins[0]][0] > rect[mins[1]][0]:
                return np.roll(rect, -mins[1], axis=0)
            else:
                return np.roll(rect, -mins[0], axis=0)
        return rect

    def mesh_from_vertices(self, polygon, line, indexes=[1, 0, 1], normal=[0, 0, 0], flip=False, texture=None,
                           color=None):
        vertices = list()
        uv = list()
        normals = list()
        faces = list()
        triangles = np.array(tripy.earclip(polygon))
        for triangle in triangles:
            triangle_points = np.zeros((3, 3))
            counter = 0
            for i, used in enumerate(indexes):
                triangle_points[0][i] = triangle[0][counter] if used else line(triangle[0][0])
                triangle_points[1][i] = triangle[1][counter] if used else line(triangle[1][0])
                triangle_points[2][i] = triangle[2][counter] if used else line(triangle[2][0])
                counter += 1 if used else 0

            vertices += triangle_points.tolist()

            # center = np.array([self.center[0], 3, self.center[1]])
            # normal_1 = (center - triangle_points[0]) / np.linalg.norm(center - triangle_points[0])
            # normal_2 = (center - triangle_points[1]) / np.linalg.norm(center - triangle_points[1])
            # normal_3 = (center - triangle_points[2]) / np.linalg.norm(center - triangle_points[2])
            # normals += [normal_1, normal_2, normal_3]
            normals += [normal, normal, normal]
            index = len(faces) * 3
            faces.append((index, index + 1, index + 2))

            if flip:
                uv += triangle[:, [1, 0]].tolist()
            else:
                uv += triangle.tolist()

        uv = np.array(uv).astype(float)
        uv.T[0] = (uv.T[0] - uv.T[0].min()) / uv.T[0].max()
        uv.T[1] = (uv.T[1] - uv.T[1].min()) / uv.T[1].max()

        # another way of applying shadows, currently for better performance
        # constant shadowing with [0, 0, 0] normals is used
        if texture is not None:
            path = os.path.join(self.path_to_textures, texture, "texture.png")
            # mtl = Material.with_texture_image(path, mode='constant', ambient=(1.0, 1.0, 1.0))
            mtl = Material.with_texture_image(path, ambient=(0.8, 0.8, 0.8))
            return TexturedMesh(vertices, normals, uv, mtl)
        else:
            color = np.array(color) / 255
            mesh = Mesh.from_faces(vertices, faces, colors=color)
            mesh._normals = normals
            return mesh

    def mesh_from_model(self, model, rect, pre_rotate=True, offset_top=False):
        path = os.path.join(self.path_to_models, model, "raw_model.obj")
        mesh = TexturedMesh.from_file(path)

        rect = self.sorted_rect(rect)
        rect_side_1 = np.linalg.norm(rect[1] - rect[0])
        rect_side_2 = np.linalg.norm(rect[2] - rect[1])

        bbox = mesh.bbox
        mesh_sizes = bbox[1] - bbox[0]
        mesh_side_1 = mesh_sizes[0]
        mesh_side_2 = mesh_sizes[2]

        if (rect_side_1 > rect_side_2) ^ (mesh_side_1 > mesh_side_2):
            if pre_rotate:
                mesh.rotate_y(np.pi / 2)

        angle = self.get_angle(rect[1] - rect[0], [1, 0])
        # mesh.rotate_y(angle)

        center = np.array(cv2.minAreaRect(rect)[0]) * self.m_in_pixel
        offset = [center[0], 0.01, center[1]]
        if offset_top:
            offset[1] = self.wall_height - mesh_sizes[1]
        mesh.offset = offset

        mesh._material.ambient = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        mesh._material.diffuse = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        mesh._material.specular = np.asarray([0.3, 0.3, 0.3], dtype=np.float32)
        mesh.scale(0.95)

        return mesh, offset, angle

    def get_angle(self, vector_1, vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        return np.arccos(dot_product)

    def sample_texture(self, style):
        ignore_style = (style is None)

        def satisfies(t):
            if not ignore_style:
                if t['style'] != style:
                    return False

            if t['category'] == "Flooring" or t['category'] == "Tile":
                return True
            return False


        filtered = list(filter(satisfies, self.texture_info))
        return random.choice(filtered)['model_id']

    def sample_models(self, category, style, x, z, y=None, count=5, threshold=0.05):
        ignore_style = (style is None)

        def satisfies(m):
            m = m[1]
            bbox = m['boundingBox']

            if m['super-category'] != category:
                return False

            if not (
                    ((bbox['xLen'] * 0.01 * (1 - threshold) < x < bbox['xLen'] * 0.01 * (1 + threshold)) and
                     (bbox['yLen'] * 0.01 * (1 - threshold) < z < bbox['yLen'] * 0.01 * (1 + threshold))) or
                    ((bbox['xLen'] * 0.01 * (1 - threshold) < z < bbox['xLen'] * 0.01 * (1 + threshold)) and
                     (bbox['yLen'] * 0.01 * (1 - threshold) < x < bbox['yLen'] * 0.01 * (1 + threshold)))
            ):
                return False

            if y is not None:
                if not (bbox['zLen'] * 0.01 * (1 - threshold) < y < bbox['zLen'] * 0.01 * (1 + threshold)):
                    return False
            else:
                if bbox['zLen'] > self.wall_height * 100:
                    return False

            if not ignore_style:
                if m['style'] != style:
                    return False

            return True

        models = list(self.models_info.items())
        filtered = list(filter(satisfies, models))
        if not filtered:
            print("    STYLE IGNORED")
            ignore_style = True
            filtered = list(filter(satisfies, models))

        if not filtered:
            filtered = list(filter(lambda x: x[1]['super-category'] == category, models))

        if not filtered:
            threshold = 0.2
            filtered = list(filter(satisfies, models))

        #         return sample(filtered, min(count, len(filtered)))
        return filtered

    def load_lighting(self, style):
        self.furniture_clusters["Lighting"].append(np.arange(len(self.bboxes["Lighting"])).tolist())

        print("\n", "Lighting", " splitted into: ", 1, " cluster(s)", sep="")
        print("    cluster size: ", len(self.bboxes["Lighting"]))
        if 0 in self.furniture_models["Lighting"]:
            model = self.furniture_models["Lighting"][0]
        else:
            models = self.sample_models("Lighting", style, 0.5, 0.5, 0.7, threshold=0.2)
            model = random.choice(models)[0]
            print("    found", len(models), "for avg sizes", 0.5, 0.5, 0.7)

        print("    using", model, "style:", style)
        for i, (_, rect) in enumerate(self.bboxes["Lighting"]):
            mesh, _, angle = self.mesh_from_model(model, rect, offset_top=True)
            self.furniture_meshes["Lighting"][i] = mesh
            self.furniture_rotations["Lighting"][i] = angle
            self.furniture_models["Lighting"][i] = model

    def load_furniture(self, category, style):
        if not self.bboxes[category]:
            return

        self.furniture_meshes[category] = dict()
        self.furniture_rotations[category] = dict()
        self.furniture_clusters[category] = list()

        if category not in self.furniture_models:
            self.furniture_models[category] = dict()

        if category == "Lighting":
            self.load_lighting(style)
            return

        points_data = []
        for _, rect in self.bboxes[category]:
            rect = cv2.minAreaRect(rect)
            sizes = np.array(rect[1]) * self.m_in_pixel
            if sizes[0] > sizes[1]:
                sizes[0], sizes[1] = sizes[1], sizes[0]
            points_data.append([sizes[0], sizes[1]])

        model = DBSCAN(eps=0.05, min_samples=1)
        yhat = model.fit_predict(points_data)
        clusters = unique(yhat)
        print("\n", category, " splitted into: ", len(clusters), " clusters", sep="")
        for cluster in clusters:
            indexes = where(yhat == cluster)
            print("    cluster size: ", len(indexes[0]))
            # calculate average size in cluster
            sizes_list = list()
            indexes = np.array(indexes)[0]
            self.furniture_clusters[category].append(indexes.tolist())
            for index in indexes:
                _, rect = self.bboxes[category][index]
                rect = cv2.minAreaRect(rect)
                sizes_list.append(np.array(rect[1]) * self.m_in_pixel)

            if indexes[0] in self.furniture_models[category]:
                model = self.furniture_models[category][indexes[0]]
            else:
                avg_x, avg_z = np.mean(np.array(sizes_list).T[0]), np.mean(np.array(sizes_list).T[1])
                models = self.sample_models(category, style, avg_x, avg_z)
                model = random.choice(models)[0]
                print("    found", len(models), "for avg sizes", avg_x, avg_z)

            print("    using", model, "style:", style)
            for index in indexes:
                _, rect = self.bboxes[category][index]
                mesh, _, angle = self.mesh_from_model(model, rect, pre_rotate=(category != 'Chair'))
                self.furniture_meshes[category][index] = mesh
                self.furniture_rotations[category][index] = angle
                self.furniture_models[category][index] = model

