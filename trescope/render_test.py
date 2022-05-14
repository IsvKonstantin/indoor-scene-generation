import json
import sys

from trescope import Trescope, Layout
from trescope.config import FRONT3DConfig, PerspectiveCamera
from trescope.toolbox import simpleFileOutputs

# UNSUPPORTED_MESH_TYPES = [
#     'SlabTop',
#     'SewerPipe',
#     'Flue',
#     '',
#     'Customized_wainscot',
#     'ExtrusionCustomizedBackgroundWall',
#     'Column',
#     'CustomizedPlatform',
#     'SlabBottom',
#     'CustomizedPersonalizedModel',
#     'SmartCustomizedCeiling',
#     'ExtrusionCustomizedCeilingModel',
#     'LightBand',
#     'CustomizedCeiling',
#     'Ceiling',
#     'SlabSide',
# ]

UNSUPPORTED_MESH_TYPES = [
"SlabTop",
  "SewerPipe",
  "CustomizedBackgroundModel",
  "Cabinet/LightBand",
  "Beam",
  "Flue",
  "",
  "CustomizedFurniture",
  "Customized_wainscot",
  "ExtrusionCustomizedBackgroundWall",
  "Column",
  "CustomizedPlatform",
  "CustomizedFixedFurniture",
  "SlabBottom",
  "CustomizedPersonalizedModel",
  "SmartCustomizedCeiling",
  "ExtrusionCustomizedCeilingModel",
  "Hole",
  "LightBand",
  "Window",
  "BayWindow",
  "CustomizedFeatureWall",
  "CustomizedCeiling",
  "Front",
  "Back",
  "Cornice",
  "Ceiling",
  "Door",
  # "Floor",
  "WallOuter",
  "SlabSide",
  "Pocket",
  "WallBottom",
  "WallTop",
  "Cabinet",
  "Baseboard",
  "WallInner"
]

file = sys.argv[1]
location = sys.argv[2]
output_dir = sys.argv[3]
description = sys.argv[4]
height = sys.argv[5]

with open(description) as json_file:
    data = json.load(json_file)

pos = [data['target'][0], float(height), data['target'][2]]
target = [data['target'][0], 0, data['target'][2]]


def visualize(output_id, the_file):
    Trescope().selectOutput(output_id).updateLayout(
        Layout().camera(
            PerspectiveCamera().up(0, 1, 0).center(*target).eye(*pos).fovy(60).near(0.1).far(1000)))

    (Trescope().selectOutput(output_id)
     .plotFRONT3D(the_file)
     .withConfig(FRONT3DConfig()
                 # .view('top')
                 .shapeLocalSource('/home/flexatroid/Diploma/Data/3D-FUTURE-model/')
                 .hiddenMeshes(UNSUPPORTED_MESH_TYPES)
                 .renderType('color')))
    Trescope().selectOutput(output_id).flush()
    Trescope().selectOutput(output_id).clear()


Trescope().initialize(True, simpleFileOutputs(output_dir, [file], 256, 256))
visualize(file, location)
