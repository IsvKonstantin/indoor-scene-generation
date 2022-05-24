"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import json
import os
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import torch

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html


opt = TestOptions().parse()
opt.status = 'UI_mode'

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))


def load_style(noise_vector):
    noise_info = json.load(open('datasets/noise_info.json'))
    noise_vector = list(map(lambda x: float(x), noise_vector))

    available = os.listdir('styles_test/style_codes/')
    filtered = []
    for room in noise_info:
        if noise_info[room] == noise_vector and room in available:
            filtered.append(room)

    room = random.choice(filtered)

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_code_folder = 'styles_test/style_codes/' + room
    print(room)

    input_style_dic = {}

    for label in range(opt.label_nc):
        input_style_dic[str(label)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(label), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(label), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                input_style_dic[str(label)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(input_style_code_folder, str(label), style_code_path + '.npy'))).cuda()

            else:
                input_style_dic[str(label)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(label), style_code_path + '.npy'))).cuda()

    return input_style_dic


furniture_string = opt.furniture
furniture_vector = furniture_string.split(" ")
print(furniture_vector)

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    data_i['obj_dic'] = load_style(furniture_vector)
    data_i['image'] = torch.zeros([1, 3, 256, 256])
    generated = model(data_i, mode='UI_mode')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
