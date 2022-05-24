import argparse
import numpy as np
from PIL import Image
from PIL import ImageColor
from rendering.render import ImageInstance
from image.utils import convert_label

parser = argparse.ArgumentParser(description='Render configuration')
parser.add_argument('--path_to_models', type=str, default='3D-FRONT-base/3D-FUTURE-model/', help='Path to 3D-FRONT scene files')
parser.add_argument('--path_to_textures', type=str, default='3D-FRONT-base/3D-FRONT-texture/', help='Where to store parsed scenes')
parser.add_argument('--path_to_image', type=str, default='examples/image.png', help='Room image')
parser.add_argument('--path_to_label', type=str, default='examples/label.png', help='Room floor label map')
parser.add_argument('--path_to_config', type=str, default=None, help='If specified, fill room with furniture from configuration file')
parser.add_argument('--save_config', action='store_true', default=False, help='If specified, save furniture info to configuration file')
parser.add_argument('--render_mode', type=str, default='default', help='"default" - free camera; "gif" - save gif; "auto" - auto camera')
parser.add_argument('--render_walls', action='store_true', default=False, help='Render walls')
parser.add_argument('--wall_color', type=str, default="#536CB5", help='Wall color, hex value')
parser.add_argument('--floor_texture', type=str, default=None, help='Floor texture from 3D-Front dataset')
parser.add_argument('--show_axes', action='store_true', default=False, help='Show axes at 0, 0, 0 coordinates')
parser.add_argument('--style', type=str, default=None, help='Style of room furniture')

opt = parser.parse_args()

if __name__ == '__main__':
    image = Image.open(opt.path_to_image)
    image = np.array(image)

    label = Image.open(opt.path_to_label)
    label = convert_label(np.array(label))

    opt.render_mode = "gif"
    opt.wall_color = ImageColor.getcolor(opt.wall_color, "RGB")

    image_instance = ImageInstance(image, label, opt)
    image_instance.process_image()
    image_instance.render_image()

