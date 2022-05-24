import numpy as np
from variables import *


def convert_label(label):
    if len(label.shape) == 2:
        return label

    label_parsed = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            if label[i][j].tolist() == COLORS[CONNECTION_COLORS['floor']]:
                label_parsed[i][j] = 1
            if label[i][j].tolist() == COLORS[CONNECTION_COLORS['door']]:
                label_parsed[i][j] = 2
            if label[i][j].tolist() == COLORS[CONNECTION_COLORS['window']]:
                label_parsed[i][j] = 3
            if label[i][j].tolist() == [0, 0, 0]:
                label_parsed[i][j] = 0

    return label_parsed

