import json
import logging
import os
import subprocess
import time

import numpy as np

from trescope import Trescope, Layout
from trescope.config import (Mesh3DConfig, PerspectiveCamera, FRONT3DConfig)
from trescope.toolbox import simpleDisplayOutputs, color_from_label, simpleFileOutputs, visualize_front3d_mesh


UNSUPPORTED_MESH_TYPES = [
"SlabTop",
  # "SewerPipe",
  # "CustomizedBackgroundModel",
  # "Cabinet/LightBand",
  # "Beam",
  # "Flue",
  # "",
  # "CustomizedFurniture",
  # "Customized_wainscot",
  # "ExtrusionCustomizedBackgroundWall",
  # "Column",
  # "CustomizedPlatform",
  # "CustomizedFixedFurniture",
  # "SlabBottom",
  # "CustomizedPersonalizedModel",
  # "SmartCustomizedCeiling",
  # "ExtrusionCustomizedCeilingModel",
  # "Hole",
  # "LightBand",
  # "Window",
  # "BayWindow",
  # "CustomizedFeatureWall",
  # "CustomizedCeiling",
  # "Front",
  # "Back",
  "Cornice",
  "Ceiling",
  # "Door",
  # "Floor",
  # "WallOuter",
  # "SlabSide",
  # "Pocket",
  # "WallBottom",
  "WallTop",
  # "Cabinet",
  # "Baseboard",
  # "WallInner"
]



front3d_base = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data/res/3D-FRONT-samples'))


def visualize_front3d_mesh_type(output_id, front3d_scene_file):
    with open(front3d_scene_file) as file:
        front3d_scene = json.load(file)
    type_cluster = {}
    for mesh in front3d_scene['mesh']:
        mesh_type = mesh['type']
        if mesh_type not in type_cluster:
            type_cluster[mesh_type] = {'xyz': np.array(mesh['xyz']).reshape((-1, 3)),
                                       'faces': np.array(mesh['faces'], dtype=np.int).reshape((-1, 3))}
        else:
            xyz = np.array(mesh['xyz']).reshape((-1, 3))
            faces = (np.array(mesh['faces'], dtype=np.int) + len(type_cluster[mesh_type]['xyz'])).reshape((-1, 3))
            type_cluster[mesh_type]['xyz'] = np.vstack((type_cluster[mesh_type]['xyz'], xyz))
            type_cluster[mesh_type]['faces'] = np.vstack((type_cluster[mesh_type]['faces'], faces))

    for index, (mesh_type, mesh) in enumerate(type_cluster.items()):
        Trescope().selectOutput(output_id).updateLayout(
            Layout().showLegend(False).camera(PerspectiveCamera().up(0, 1, 0).eye(0, 2.3, 0)))
        (Trescope()
         .selectOutput(output_id)
         .plotMesh3D(*mesh['xyz'].T)
         .withConfig(Mesh3DConfig().indices(*mesh['faces'].T).color(color_from_label(index)).name(mesh_type)))
    Trescope().selectOutput(output_id).flush()


def visualize_front3d_color(output_id, front3d_scene_file):
    # "dc74ad5c-34cf-4237-a3f3-a94297a907c8_21.jpg": {
    #     "scene_file": "data/camera_pos/dc74ad5c-34cf-4237-a3f3-a94297a907c8.json",
    #     "camera": {
    #         "aspect": 1.5,
    #         "far": 1000,
    #         "fov": 98,
    #         "near": 0.1,
    #         "pos": [
    #             2.4812,
    #             1.2889963074424384,
    #             1.4702601021056108
    #         ],
    #         "target": [
    #             1.7581605268137603,
    #             1.0697053147058584,
    #             1.4628596111470564
    #         ],
    #         "up": [
    #             0,
    #             1,
    #             0
    #         ]
    #     }
    # },

    # with open('camera_data.json') as f:
    #     camera = json.load(f)[0]
    #

    camera = {"pos": [0.5, 9.35, 1], "target": [0.5, 0, 1], "fov": 60}

    Trescope().selectOutput(output_id).updateLayout(
        Layout().camera(
            PerspectiveCamera().up(0, 1, 0).center(*camera['target']).eye(*camera['pos']).fovy(camera['fov']).near(
                0.1).far(1000)))

    (Trescope().selectOutput(output_id)
     .plotFRONT3D(front3d_scene_file)
     .withConfig(FRONT3DConfig()
                 .view('top')
                 # .shapeLocalSource('/home/flexatroid/Diploma/Python/Testing/3D-FRONT-samples/3D-FUTURE-model')
                 .shapeLocalSource('/home/flexatroid/Diploma/Data/3D-FUTURE-model/')
                 # .shapeLocalSource(os.path.join(front3d_base, '3D-FUTURE-model'))
                 .hiddenMeshes(UNSUPPORTED_MESH_TYPES)
                 .renderType('color')))
    Trescope().selectOutput(output_id).flush()


def visualize_front3d_depth(output_id, front3d_scene_file):
    (Trescope().selectOutput(output_id)
     .plotFRONT3D(front3d_scene_file)
     .withConfig(FRONT3DConfig()
                 .view('top')
                 .shapeLocalSource(os.path.join(front3d_base, '3D-FUTURE-model'))
                 .hiddenMeshes(['Ceiling', 'CustomizedCeiling'])
                 .renderType('depth')))
    Trescope().selectOutput(output_id).flush()


def visualize_front3d_normal(output_id, front3d_scene_file):
    (Trescope().selectOutput(output_id)
     .plotFRONT3D(front3d_scene_file)
     .withConfig(FRONT3DConfig()
                 .view('top')
                 .shapeLocalSource(os.path.join(front3d_base, '3D-FUTURE-model'))
                 .hiddenMeshes(['Ceiling', 'CustomizedCeiling'])
                 .renderType('normal')))
    Trescope().selectOutput(output_id).flush()


def main(output_type):
    front3d_scene = os.path.join(front3d_base, '3D-FRONT')

    # output_ids = ['main', 'room1', 'room2', 'room3']
    output_ids = ['preview']

    if 'file' == output_type:
        Trescope().initialize(True, simpleFileOutputs('gen', output_ids, 720, 720))
    else:
        Trescope().initialize(True, simpleDisplayOutputs(1, 1, output_ids))

    # visualize_front3d_color('preview', '/home/flexatroid/Diploma/Python/Testing/preview/25b989a3-3bc5-45fb-93a9-7d9c52217e15_LivingDiningRoom-12141.json')

    # visualize_front3d_color('preview', '/home/flexatroid/Diploma/Python/Testing/preview/3d24a08b-c20e-4cf3-b0ba-ea4d4a06bfda.json')
    # c38b86ae-e6c5-4a1f-8b52-eceb83456d80

    visualize_front3d_color('preview', '/home/flexatroid/Diploma/Python/Testing/preview/c38b86ae-e6c5-4a1f-8b52-eceb83456d80.json')

    # visualize_front3d_color('main', '/home/flexatroid/Diploma/Python/Testing/hahaha/main.json')
    # visualize_front3d_color('room1', '/home/flexatroid/Diploma/Python/Testing/hahaha/room1.json')
    # visualize_front3d_color('room2', '/home/flexatroid/Diploma/Python/Testing/hahaha/room2.json')
    # visualize_front3d_color('room3', '/home/flexatroid/Diploma/Python/Testing/hahaha/room3.json')

    # visualize_front3d_color('preview', '/home/flexatroid/Diploma/Python/Testing/preview/25b989a3-3bc5-45fb-93a9-7d9c52217e15.json')
    # visualize_front3d_color('filtered', '/home/flexatroid/Diploma/Python/Testing/preview/preview-filtered-4.json')


if __name__ == '__main__':
    # os.system("pkill trescope")

    # try:
    #     utils.texture_color_mask(
    #         '/home/flexatroid/Diploma/Python/Testing/3D-FRONT-samples/3D-FUTURE-model/6934dea0-1d66-49c4-82c6-4d54d41f9707/texture.png',
    #         'newt.png', utils.colors['red'])
    # except Exception as e:
    #     logging.error("Pic trash! bad =)")

    # utils.modify_textures('/home/flexatroid/Diploma/Data/3D-FUTURE-model/',
    #                       'info/model_info.json')

    # split.split_room('/home/flexatroid/Diploma/Python/Testing/3D-FRONT-samples/output',
    #                  '/home/flexatroid/Diploma/Python/Testing/3D-FRONT-samples')
    main('view')
