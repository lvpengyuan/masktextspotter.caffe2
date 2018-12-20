# Modified by Minghui Liao and Pengyuan Lyu
###############################################################################
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

from caffe2.python import workspace

from core.config import cfg
from core.config import get_output_dir
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.text_dataset import TextDataSet
from modeling import model_builder
from utils.io import save_object
from utils.timer import Timer
import utils.c2 as c2_utils
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils

logger = logging.getLogger(__name__)


def test_net_on_dataset(multi_gpu=False, vis=False):
    """Run inference on a dataset."""
    output_dir = get_output_dir(training=False)
    dataset = TextDataSet(cfg.TEST.DATASET)
    test_timer = Timer()
    test_timer.tic()
    # if multi_gpu:
    #     num_images = len(dataset.get_roidb())
    #     all_boxes, all_global_segms, all_char_segms, all_keyps = multi_gpu_test_net_on_dataset(
    #         num_images, output_dir
    #     )
    # else:
    test_net(vis=vis)
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))



def multi_gpu_test_net_on_dataset(num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, 'test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(ind_range=None, vis=False):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert cfg.TEST.WEIGHTS != '', \
        'TEST.WEIGHTS must be set to the model file to test'
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'
    assert cfg.TEST.DATASET != '', \
        'TEST.DATASET must be set to the dataset name to test'

    output_dir = get_output_dir(training=False)
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        ind_range
    )
    model = initialize_model_from_cfg()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_keyps, all_polygons, all_strs = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        if cfg.MODEL.FASTER_RCNN:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue

        im = cv2.imread(entry['image'])
        image_name = entry['image'].split('/')[-1].split('.')[0]
        with c2_utils.NamedCudaScope(0):
            im_detect_all(
                model, im, image_name, box_proposals, timers, vis=vis
            )


def initialize_model_from_cfg():
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False)
    net_utils.initialize_from_weights_file(
        model, cfg.TEST.WEIGHTS, broadcast=False
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    return model


def get_roidb_and_dataset(ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = TextDataSet(cfg.TEST.DATASET)
    if cfg.MODEL.FASTER_RCNN:
        roidb = dataset.get_roidb()
    else:
        roidb = dataset.get_roidb(
            proposal_file=cfg.TEST.PROPOSAL_FILE,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_polygons = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    # all_global_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    # all_char_segms = [[[] for _ in range(num_images)] for _ in range(37)]
    all_strs = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_keyps, all_polygons, all_strs


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
