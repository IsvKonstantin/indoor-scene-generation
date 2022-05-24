import argparse
import os
import shutil
import subprocess
import glob
from pathlib import Path
import cv2
import numpy as np
from image.utils import convert_label
from PIL import Image

parser = argparse.ArgumentParser(description='Generate room image based on a label map (floor)')
parser.add_argument('--path_to_label', type=str, default='examples/test-label.png', help='Where to store generated images')
parser.add_argument('--path_to_output', type=str, default='examples/', help='Where to sore generated label maps')
parser.add_argument('--bed', action='store_true', default=False, help='If specified, generated room will contain bed')
parser.add_argument('--chair', action='store_true', default=False, help='If specified, generated room will contain chair')
parser.add_argument('--lighting', action='store_true', default=False, help='If specified, generated room will contain lighting')
parser.add_argument('--table', action='store_true', default=False, help='If specified, generated room will contain table')
parser.add_argument('--cabinet', action='store_true', default=False, help='If specified, generated room will contain cabinet/shelf/desk')
parser.add_argument('--sofa', action='store_true', default=False, help='If specified, generated room will contain sofa')

opt = parser.parse_args()

if __name__ == '__main__':
    Path("sean/datasets/generate_label").mkdir(parents=True, exist_ok=True)
    Path("sean/datasets/generate_img").mkdir(parents=True, exist_ok=True)

    files = glob.glob("sean/datasets/generate_label/*")
    for f in files:
        print(f)
        os.remove(f)

    files = glob.glob("sean/datasets/generate_img/*")
    for f in files:
        os.remove(f)

    dummy_image = np.zeros((256, 256, 3))
    label = Image.open(opt.path_to_label)
    label = convert_label(np.array(label))

    cv2.imwrite("sean/datasets/generate_img/generated.png", dummy_image)
    cv2.imwrite("sean/datasets/generate_label/generated.png", label)

    furniture_string = ""

    if opt.lighting:
        furniture_string += "1 "
    else:
        furniture_string += "0 "

    if opt.chair:
        furniture_string += "2 "
    else:
        furniture_string += "0 "

    if opt.table:
        furniture_string += "3 "
    else:
        furniture_string += "0 "

    if opt.cabinet:
        furniture_string += "4 "
    else:
        furniture_string += "0 "

    if opt.sofa:
        furniture_string += "5 "
    else:
        furniture_string += "0 "

    if opt.bed:
        furniture_string += "6"
    else:
        furniture_string += "0"

    subprocess.call([
        "python",
        "test_style.py",
        "--name", "sean-args",
        "--ngf", "32",
        "--load_size", "256",
        "--crop_size", "256",
        "--dataset_mode", "custom",
        "--label_dir", "datasets/generate_label",
        "--image_dir", "datasets/generate_img",
        "--label_nc", "4",
        "--no_instance",
        "--gpu_ids", "0",
        "--furniture", furniture_string
    ], cwd="sean/")

    shutil.move("sean/results/sean-args/test_latest/images/synthesized_image/generated.png", "examples/generated.png")
    shutil.rmtree("sean/results/", ignore_errors=False, onerror=None)