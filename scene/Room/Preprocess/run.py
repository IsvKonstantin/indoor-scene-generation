# -*- coding: utf-8 -*
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from floorplan_generator import FloorplanGenerator


def scene_to_floorplan(filename):

    houseInfo = {}
    floorplan_generator = FloorplanGenerator()
    houseInfo = floorplan_generator.generate_floorplan(filename)
    # print(houseInfo)
    return houseInfo

# scene_to_floorplan("/home/flexatroid/Diploma/Data/3D-FRONT/c38b86ae-e6c5-4a1f-8b52-eceb83456d80.json")