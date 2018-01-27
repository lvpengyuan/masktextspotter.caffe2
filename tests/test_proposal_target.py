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
from datasets.roidb_text import combined_roidb_for_training
from modeling import model_builder
from modeling.detector import DetectionModelHelper
from utils.logging import log_json_stats
from utils.logging import setup_logging
from utils.logging import SmoothedValue
from utils.timer import Timer
import utils.c2
import utils.env as envu
import utils.net as nu

from ops.collect_and_distribute_fpn_rpn_proposals_rec \
    import CollectAndDistributeFpnRpnProposalsRecOp
from caffe2.python import core
import roi_data.fast_rcnn
import utils.blob as blob_utils

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
		img_draw.polygon(list(polygons[i]), outline=color)
		choose_cboxes = charboxes[np.where(charboxes[:, -1] == i)[0], :]
		for j in range(choose_cboxes.shape[0]):
			img_draw.polygon(list(choose_cboxes[j][:8]), outline=color)
			char = lex[int(choose_cboxes[j][8])]
			img_draw.text(list(choose_cboxes[j][:2]), char)

	img.save('./tests/vis_dataset_icdar/' + img_path.strip().split('/')[-1])


def CollectAndDistributeFpnRpnProposalsRec(roidb, im_info):
        """Merge RPN proposals generated at multiple FPN levels and then
        distribute those proposals to their appropriate FPN levels. An anchor
        at one FPN level may predict an RoI that will map to another level,
        hence the need to redistribute the proposals.

        This function assumes standard blob names for input and output blobs.

        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
          - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
            documentation from GenerateProposals.
          - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
            level i; see rpn_roi_probs documentation from GenerateProposals.

        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).

        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max + 1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max + 1)
        ]
        blobs_in = rois_names + score_names
        blobs_in += ['roidb', 'im_info']
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsRecOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=True
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]
        workspace.ResetWorkspace()
        net = core.Net("tutorial")
        net.Python(CollectAndDistributeFpnRpnProposalsRecOp(True).forward)(blobs_in, blobs_out)
        rois = np.ones((512,5))*5
        for i in range(512):
        	rois[i,0]=i

        for roi_name in rois_names:
        	workspace.FeedBlob(roi_name, rois)
        for score_name in score_names:
        	workspace.FeedBlob(score_name, np.ones((512,1)) * 0.5)
        roidb_blob = blob_utils.serialize(roidb)
        workspace.FeedBlob('roidb', roidb_blob)
        workspace.FeedBlob('im_info', im_info)
        workspace.RunNetOnce(net)



def main(opts):
    logger = logging.getLogger(__name__)
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    # for i in range(50):
    # 	entry = roidb[random.randint(0, len(roidb))]
    # 	vis_roidb(entry)
    # entry = roidb[random.randint(0, len(roidb))]
    im_height = 720
    im_width = 1280
    im_scale = 1
    im_info = np.zeros((len(roidb), 3))
    im_info_example = np.array([im_height, im_width, im_scale])
    for i in range(len(roidb)):
    	im_info[i,:]=im_info_example
    CollectAndDistributeFpnRpnProposalsRec(roidb, im_info)
    



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