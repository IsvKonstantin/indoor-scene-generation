import json
import logging
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from house import HouseInstance

parser = argparse.ArgumentParser(description='Create room images and label maps')
parser.add_argument('--path_to_images', type=str, default='gen-images/', help='Where to store generated images')
parser.add_argument('--path_to_labels', type=str, default='gen-labels/', help='Where to sore generated label maps')
parser.add_argument('--path_to_houses', type=str, default='3D-FRONT-base/3D-FRONT-parsed/', help='Path to parsed 3D-FRONT houses')

opt = parser.parse_args()

if __name__ == '__main__':
    Path("logs/").mkdir(parents=True, exist_ok=True)
    Path(opt.path_to_images).mkdir(parents=True, exist_ok=True)
    Path(opt.path_to_labels).mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler('logs/logfile.txt', 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    for file in tqdm(os.listdir(opt.path_to_houses)):
        try:
            house_path = os.path.join(opt.path_to_houses, file)
            house = json.load(open(house_path))
            house_instance = HouseInstance(house, file[:-12], logger)
            house_instance.process_rooms()
            house_instance.save_images(opt.path_to_images, opt.path_to_labels)
        except Exception as e:
            logger.error("Exception caught / Assertion failed for " + file)

