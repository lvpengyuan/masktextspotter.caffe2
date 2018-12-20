# Modified by Minghui Liao and Pengyuan Lyu
# 
##############################################################################
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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Construct minibatches for Detectron networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np

from core.config import cfg
import roi_data.fast_rcnn
import roi_data.retinanet
import roi_data.rpn
import utils.blob as blob_utils

import random
from shapely.geometry import box, Polygon 
from shapely import affinity
import math
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        blob_names += roi_data.retinanet.get_retinanet_blob_names(
            is_training=is_training
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    if not cfg.IMAGE.aug:
        im_blob, im_scales = _get_image_blob(roidb)
        blobs['data'] = im_blob
        if cfg.RPN.RPN_ON:
            # RPN-only or end-to-end Faster/Mask R-CNN
            valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
        elif cfg.RETINANET.RETINANET_ON:
            im_width, im_height = im_blob.shape[3], im_blob.shape[2]
            # im_width, im_height corresponds to the network input: padded image
            # (if needed) width and height. We pass it as input and slice the data
            # accordingly so that we don't need to use SampleAsOp
            valid = roi_data.retinanet.add_retinanet_blobs(
                blobs, im_scales, roidb, im_width, im_height
            )
        else:
            # Fast R-CNN like models trained on precomputed proposals
            valid = roi_data.fast_rcnn.add_fast_rcnn_blobs_rec(blobs, im_scales, roidb)
        return blobs, valid
    else:
        im_blob, im_scales, new_roidb = _get_image_aug_blob(roidb)
        blobs['data'] = im_blob
        if cfg.RPN.RPN_ON:
            # RPN-only or end-to-end Faster/Mask R-CNN
            valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, new_roidb)
        elif cfg.RETINANET.RETINANET_ON:
            im_width, im_height = im_blob.shape[3], im_blob.shape[2]
            # im_width, im_height corresponds to the network input: padded image
            # (if needed) width and height. We pass it as input and slice the data
            # accordingly so that we don't need to use SampleAsOp
            valid = roi_data.retinanet.add_retinanet_blobs(
                blobs, im_scales, new_roidb, im_width, im_height
            )
        else:
            # Fast R-CNN like models trained on precomputed proposals
            valid = roi_data.fast_rcnn.add_fast_rcnn_blobs_rec(blobs, im_scales, new_roidb)
        return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def _get_image_aug_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im = im.astype(np.float)
        
        new_rec = roi_rec.copy()
        boxes = roi_rec['boxes'].copy()
        polygons = roi_rec['segms']
        charboxes = roi_rec['charboxes'].copy()
        segms = roi_rec['segms']

        if cfg.IMAGE.aug:
            im, boxes, polygons, charboxes = _rotate_image(im, boxes, polygons, charboxes)
            im = _random_saturation(im)
            im = _random_hue(im)
            im = _random_lighting_noise(im)
            im = _random_contrast(im)
            im = _random_brightness(im)

        im_info_height = im.shape[0]
        im_info_width = im.shape[1]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE
        )
        im=im[0]
        im_scale=im_scale[0]
        processed_ims.append(im)
        new_rec['height'] = im_info_height
        new_rec['width'] = im_info_width
        im_info = [im_info_height, im_info_width, im_scale]
        new_rec['boxes'] = boxes
        # new_rec['polygons'] = polygons
        # for j, polygon in enumerate(new_rec['polygons']):
        #     segms[j] = [[polygon[0], polygon[1], polygon[2], polygon[3], polygon[4], polygon[5], polygon[6], polygon[7]]]
        new_rec['segms'] = polygons
        new_rec['charboxes'] = charboxes
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
        im_scales.append(im_scale)
        # new_rec['height'] = new_rec['height'] * im_scale
        # new_rec['width'] = new_rec['width'] * im_scale
        # new_rec['boxes'] = _clip_boxes(np.round(boxes.copy() * im_scale), im_info[:2])
        # new_rec['polygons'] = _clip_polygons(np.round(polygons.copy() * im_scale), im_info[:2])
        # for j, polygon in enumerate(new_rec['polygons']):
        #     segms[j] = [[polygon[0], polygon[1], polygon[2], polygon[3], polygon[4], polygon[5], polygon[6], polygon[7]]]
        # new_rec['segms'] = segms
        # new_rec['charboxes'] = _resize_clip_char_boxes(charboxes.copy(), im_scale, im_info[:2])
        # new_rec['im_info'] = im_info
        # processed_roidb.append(new_rec)
        # im_scales.append(1.0)
        # polygons_scale = []
        # for polygon in polygons:
        #     polygon_scale = list(np.array(polygon[0])*im_scale)
        #     polygons_scale.append([polygon_scale])
        # name = roidb[i]['image'].split('/')[-1]
        # _visualize_roidb(im, np.round(boxes.copy() * im_scale), polygons_scale, charboxes*im_scale, name)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales, processed_roidb

def _clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def _clip_polygons(polygons, im_shape):
    """
    Clip polygons to image boundaries.
    :param polygons: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    polygons[:, 0:8:2] = np.maximum(np.minimum(polygons[:, 0:8:2], im_shape[1] - 1), 0)
    # y1 >= 0
    polygons[:, 1:8:2] = np.maximum(np.minimum(polygons[:, 1:8:2], im_shape[0] - 1), 0)
    return polygons

def _resize_clip_char_boxes(polygons, im_scale, im_shape):
    """
    Clip polygons to image boundaries.
    :param polygons: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    polygons[:, :8] = np.round(polygons[:, :8]*im_scale)
    polygons[:, 0:8:2] = np.maximum(np.minimum(polygons[:, 0:8:2], im_shape[1] - 1), 0)
    # y1 >= 0
    polygons[:, 1:8:2] = np.maximum(np.minimum(polygons[:, 1:8:2], im_shape[0] - 1), 0)
    return polygons

def _rotate_image(im, boxes, polygons, charboxes):
    if cfg.IMAGE: 
        delta = cfg.IMAGE.rotate_delta
        prob = cfg.IMAGE.rotate_prob
    else:
        delta = 10
        prob = 1
    new_boxes = boxes.copy()
    new_polygons = polygons
    new_charboxes = charboxes.copy()
    if random.random() < prob:
        delta = random.uniform(-1*delta, delta)
        ## rotate image first
        height, width, _ = im.shape
        ## get the minimal rect to cover the rotated image
        img_box = np.array([[0, 0, width, 0, width, height, 0, height]])
        rotated_img_box = _quad2minrect(_rotate_polygons(img_box, -1*delta, (width/2, height/2)))
        r_height = int(max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(rotated_img_box[0][3], rotated_img_box[0][1]))
        r_width = int(max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(rotated_img_box[0][2], rotated_img_box[0][0]))

        ## padding im
        im_padding = np.zeros((r_height, r_width, 3))
        start_h, start_w = int((r_height - height)/2.0), int((r_width - width)/2.0)
        end_h, end_w = start_h + height, start_w + width
        im_padding[start_h:end_h, start_w:end_w, :] = im
        ## get new boxes, polygons, charboxes
        boxes[:, 0::2] += start_w
        boxes[:, 1::2] += start_h
        # polygons[:, ::2] += start_w
        # polygons[:, 1::2] += start_h
        charboxes[:, :8:2] += start_w
        charboxes[:, 1:8:2] += start_h

        M = cv2.getRotationMatrix2D((r_width/2, r_height/2), delta, 1)
        im = cv2.warpAffine(im_padding, M, (r_width, r_height))
        
        ## polygons
        new_polygons = _rotate_segms(polygons, -1*delta, (r_width/2, r_height/2), start_h, start_w)
        new_boxes[:, :4] = _quadlist2minrect(new_polygons)
        ## charboxes
        if charboxes[:, -1].mean() != -1:
            new_charboxes[:, :8] = _rotate_polygons(charboxes[:, :8], -1*delta, (r_width/2, r_height/2))
    return im, new_boxes, new_polygons, new_charboxes

def _rotate_polygons(polygons, angle, r_c):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15

    poly_list = _quad2boxlist(polygons)
    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords))<5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = _boxlist2quads(rotate_boxes_list)
    return res

def _rotate_segms(polygons, angle, r_c, start_h, start_w):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15
    poly_list=[]
    for polygon in polygons:
        tmp=[]
        for i in range(int(len(polygon[0]) / 2)):
            tmp.append([polygon[0][2*i] + start_w, polygon[0][2*i+1] + start_h])
        poly_list.append(tmp)

    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords))<5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = []
    for i, box in enumerate(rotate_boxes_list):
        tmp = []
        for point in box:
            tmp.append(point[0])
            tmp.append(point[1])
        res.append([tmp])

    return res

def _random_saturation(im):
    if cfg.IMAGE:
        prob = cfg.IMAGE.saturation_prob
        lower = cfg.IMAGE.saturation_lower
        upper = cfg.IMAGE.saturation_upper
    else:
        prob = 0.5
        lower = 0.5
        upper = 1.5

    assert upper >= lower, "saturation upper must be >= lower."
    assert lower >= 0, "saturation lower must be non-negative."
    if random.random() < prob:
        im[:, :, 1] *= random.uniform(lower, upper)
    return im

def _random_hue(im):
    if cfg.IMAGE:
        prob = cfg.IMAGE.hue_prob
        delta = cfg.IMAGE.hue_delta
    else:
        prob = 0.5
        delta = 18
    if random.random() < prob:
        im[:, :, 0] += random.uniform(delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
    return im

def _random_lighting_noise(im):
    if cfg.IMAGE:
        prob = cfg.IMAGE.lighting_noise_prob
    else:
        prob = 0.5
    perms = ((0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0))

    if random.random() < prob:
        swap = perms[random.randint(0, len(perms) - 1)]
        im = im[:, :, swap]
    return im

def _random_contrast(im):
    if cfg.IMAGE:
        prob = cfg.IMAGE.contrast_prob
        lower = cfg.IMAGE.contrast_lower
        upper = cfg.IMAGE.contrast_upper
    else:
        prob = 0.5
        lower = 0.5
        upper = 1.5
    assert upper >= lower, "contrast upper must be >= lower."
    assert lower >= 0, "contrast lower must be non-negative."
    if random.random() < prob:
        alpha = random.uniform(lower, upper)
        im *= alpha
    return im

    

def _random_brightness(im):
    if cfg.IMAGE:
        delta = cfg.IMAGE.brightness_delta
        prob = cfg.IMAGE.brightness_prob
    else:
        delta = 32
        prob = 0.5
    if random.random() < prob:
        delta = random.uniform(-1*delta, delta)
        im += delta
    return im

def _rect2quad(boxes):
    x_min, y_min, x_max, y_max = boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 2].reshape((-1, 1)), boxes[:, 3].reshape((-1, 1))
    return np.hstack((x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max))

def _quad2rect(boxes):
    ## only support rectangle
    return np.hstack((boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 4].reshape((-1, 1)), boxes[:, 5].reshape((-1, 1))))

def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)), boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))


def _quad2boxlist(boxes):
    res = []
    for i in range(boxes.shape[0]):
        res.append([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]], [boxes[i][4], boxes[i][5]], [boxes[i][6], boxes[i][7]]])
    return res

def _boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
    return res

def _visualize_roidb(im, boxes, polygons, charboxes, name):
    lex = '_0123456789abcdefghijklmnopqrstuvwxyz'
    img = np.array(im, dtype=np.uint8)
    img = Image.fromarray(img)
    img_draw = ImageDraw.Draw(img)
    for i in range(boxes.shape[0]):
        color = _random_color()
        img_draw.rectangle(list(boxes[i][:4]), outline=color)
        img_draw.polygon(polygons[i][0], outline=color)
        choose_cboxes = charboxes[np.where(charboxes[:, -1] == i)[0], :]
        for j in range(choose_cboxes.shape[0]):
            img_draw.polygon(list(choose_cboxes[j][:8]), outline=color)
            # char = lex[int(choose_cboxes[j][8])]
            # img_draw.text(list(choose_cboxes[j][:2]), char)

    img.save('./tests/visu_data_aug/' + name)
    print('image saved')
    raw_input()

def _random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (r, g, b)
