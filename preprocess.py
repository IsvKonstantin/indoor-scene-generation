import argparse
import json
import os
from tqdm import tqdm

from scene.raw_room import RawRoomInstance

parser = argparse.ArgumentParser(description='Parse 3D-FRONT dataset')
parser.add_argument('--input', type=str, default='3D-FRONT-base/3D-Front/', help='Path to 3D-FRONT scene files')
parser.add_argument('--output', type=str, default='3D-FRONT-base/3D-FRONT-parsed/', help='Where to store parsed scenes')
args = parser.parse_args()

if __name__ == '__main__':
    errors = []

    for file in tqdm(os.listdir(args.input)):
        try:
            scene_file = os.path.join(args.input, file)
            raw_room_instance = RawRoomInstance(scene_file)
            raw_room_instance.parse(args.output)
        except Exception as e:
            errors.append(file)

    with open('errors.json', 'w') as outfile:
        json.dump(errors, outfile, indent=None)

