from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import datetime
import logging
import numpy as np
import os
import pprint
import re
import sys
from PIL import Image, ImageDraw
import random
# import test_net

from caffe2.python import memonger
from caffe2.python import utils as c2_py_utils
from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import get_output_dir
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from datasets.roidb_text import combined_roidb_for_training, mix_roidbs_for_training
from modeling import model_builder
from utils.logging import log_json_stats
from utils.logging import setup_logging
from utils.logging import SmoothedValue
from utils.timer import Timer
import utils.c2
import utils.env as envu
import utils.net as nu

utils.c2.import_contrib_ops()
utils.c2.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (r, g, b)

def vis_roidb(roidb):
    img_path = roidb['image']
    flip = roidb['flipped']
    boxes = roidb['boxes']
    polygons = roidb['polygons']
    charboxes = roidb['charboxes']
    lex = '_0123456789abcdefghijklmnopqrstuvwxyz'
    img = Image.open(img_path)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_draw = ImageDraw.Draw(img)
    for i in range(boxes.shape[0]):
        color = random_color()
        img_draw.rectangle(list(boxes[i][:4]), outline=color)
        color = random_color()
        img_draw.polygon(list(polygons[i]), outline=color)
        color = random_color()
        choose_cboxes = charboxes[np.where(charboxes[:, -1] == i)[0], :]
        for j in range(choose_cboxes.shape[0]):
            img_draw.polygon(list(choose_cboxes[j][:8]), outline=color)
            char = lex[int(choose_cboxes[j][8])]
            img_draw.text(list(choose_cboxes[j][:2]), char)

    img.save('./tests/vis_dataset_total/' + img_path.strip().split('/')[-1])

    


def main(opts):
    logger = logging.getLogger(__name__)
    # # test combined_roidb_for_training
    # roidb = combined_roidb_for_training(
    #     cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    # for i in range(50):
    #   entry = roidb[random.randint(0, len(roidb))]
    #   vis_roidb(entry)
    ## test
    roidbs = mix_roidbs_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES, cfg.TRAIN.USE_CHARANNS)
    for roibd in roidbs:
        for i in range(50):
            entry = roibd[random.randint(0, len(roibd)-1)]
            vis_roidb(entry)
        raw_input()



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = utils.logging.setup_logging(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('roi_data.loader').setLevel(logging.INFO)
    np.random.seed(cfg.RNG_SEED)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Running with config:')
    logger.info(pprint.pformat(cfg))
    main(args)