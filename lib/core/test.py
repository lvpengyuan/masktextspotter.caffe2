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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Inference functionality for most Detectron models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import logging
import numpy as np

from caffe2.python import core
from caffe2.python import workspace
import pycocotools.mask as mask_util

from core.config import cfg
from utils.timer import Timer
import modeling.FPN as fpn
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
import lanms

from PIL import Image, ImageDraw, ImageFont
import os

logger = logging.getLogger(__name__)


def im_detect_all(model, im, image_name, box_proposals, timers=None, vis=False):
    print(image_name)
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scales = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scales = im_detect_bbox(model, im, box_proposals)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()
    result_logs = []
    model_path = cfg.TEST.WEIGHTS
    model_name = model_path.split('/')[-1]
    model_dir = model_path[0:len(model_path)-len(model_name)]
    save_dir_res = os.path.join(model_dir, cfg.TEST.DATASETS[0], model_name+'_results')
    
    if not os.path.isdir(save_dir_res):
        os.makedirs(save_dir_res)
    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            global_masks, char_masks, char_boxes = im_detect_mask_aug(model, im, boxes)
        else:
            global_masks, char_masks, char_boxes = im_detect_mask(model, im_scales, boxes)
        timers['im_detect_mask'].toc()
        scale = im_scales[0]
        if vis:
            img_char = np.zeros((im.shape[0], im.shape[1]))
            img_poly = np.zeros((im.shape[0], im.shape[1]))
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        for index in range(global_masks.shape[0]):
            box = boxes[index]
            box = map(int, box)
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            cls_polys = (global_masks[index, 0, :, :]*255).astype(np.uint8)
            poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
            poly_map = poly_map.astype(np.float32) / 255
            poly_map=cv2.GaussianBlur(poly_map,(3,3),sigmaX=3)
            ret, poly_map = cv2.threshold(poly_map,0.5,1,cv2.THRESH_BINARY)
            if cfg.TEST.OUTPUT_POLYGON:
                SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                poly_map = cv2.erode(poly_map,SE1) 
                poly_map = cv2.dilate(poly_map,SE1);
                poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
                im2,contours,hierarchy = cv2.findContours((poly_map*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                max_area=0
                max_cnt = contours[0]
                for cnt in contours:
                    area=cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        max_cnt = cnt
                perimeter = cv2.arcLength(max_cnt,True)
                epsilon = 0.01*cv2.arcLength(max_cnt,True)
                approx = cv2.approxPolyDP(max_cnt,epsilon,True)
                pts = approx.reshape((-1,2))
                pts[:,0] = pts[:,0] + box[0]
                pts[:,1] = pts[:,1] + box[1]
                segms = list(pts.reshape((-1,)))
                segms = map(int, segms)
                if len(segms)<6:
                    continue     
            else:      
                SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                poly_map = cv2.erode(poly_map,SE1) 
                poly_map = cv2.dilate(poly_map,SE1);
                poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
                idy,idx=np.where(poly_map == 1)
                xy=np.vstack((idx,idy))
                xy=np.transpose(xy)
                hull = cv2.convexHull(xy, clockwise=True)
                #reverse order of points.
                if  hull is None:
                    continue
                hull=hull[::-1]
                # print(hull)
                #find minimum area bounding box.
                rect = cv2.minAreaRect(hull)
                corners = cv2.boxPoints(rect)
                corners = np.array(corners, dtype="int")
                pts = get_tight_rect(corners, box[0], box[1], im.shape[0], im.shape[1], 1)
                pts_origin = [x * 1.0 for x in pts]
                pts_origin = map(int, pts_origin)
            text, rec_score, rec_char_scores = getstr_grid(char_masks[index,:,:,:].copy(), box_w, box_h)
            if cfg.TEST.OUTPUT_POLYGON:
                result_log = [int(x * 1.0) for x in box[:4]] + segms + [text] + [scores[index]] + [rec_score] + [rec_char_scores] +[len(segms)]
            else:
                result_log = [int(x * 1.0) for x in box[:4]] + pts_origin + [text] + [scores[index]] + [rec_score] + [rec_char_scores]
            result_logs.append(result_log)
            if vis:    
                if cfg.TEST.OUTPUT_POLYGON:
                    cv2.polylines(im, [np.array(segms).reshape((-1,2)).astype(np.int32)], True, color=(0, 255, 0), thickness=5)
                    # img_draw.polygon(segms, outline=(0, 255, 0))
                else:
                    img_draw.polygon(pts, outline=(0, 255, 0))
                poly = np.array(Image.fromarray(cls_polys).resize((box_w, box_h))) 
                cls_chars = 255 - (char_masks[index, 0, :, :]*255).astype(np.uint8)      
                char = np.array(Image.fromarray(cls_chars).resize((box_w, box_h)))
                img_poly[box[1]:box[3], box[0]:box[2]] = poly
                img_char[box[1]:box[3], box[0]:box[2]] = char
        
        if vis:
            save_dir_visu = os.path.join(model_dir, model_name+'_visu')
            if not os.path.isdir(save_dir_visu):
                os.mkdir(save_dir_visu)
            img_char = Image.fromarray(img_char).convert('RGB')
            img = Image.fromarray(im).convert('RGB')
            Image.blend(img, img_char, 0.5).save(os.path.join(save_dir_visu, str(image_name) + '_blend_char.jpg'))

    format_output(save_dir_res, result_logs, image_name)


def format_output(out_dir, boxes, img_name):
    res = open(os.path.join(out_dir, 'res_' + img_name.split('.')[0] + '.txt'), 'w')
    ## char score save dir
    ssur_name = os.path.join(out_dir, 'res_' + img_name.split('.')[0])
    for i, box in enumerate(boxes):
        save_name = ssur_name + '_' + str(i) + '.mat'
        if cfg.TEST.OUTPUT_POLYGON:
            np.save(save_name, box[-2])
            box = ','.join([str(x) for x in box[:4]]) + ';' + ','.join([str(x) for x in box[4:4+int(box[-1])]]) + ';' + ','.join([str(x) for x in box[4+int(box[-1]):-2]]) + ',' + save_name
        else:
            np.save(save_name, box[-1])
            box = ','.join([str(x) for x in box[:-1]]) + ',' + save_name
        # print(box)
        res.write(box + '\n')
    res.close()

def im_conv_body_only(model, im):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale_factors = _get_image_blob(im)
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale_factors


def im_detect_bbox(model, im, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        assert len(im_scales) == 1, \
            'Only single-image / single-scale batch implemented'
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
        )
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scales


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _im_scales_hf = im_detect_bbox_hflip(
            model, im, box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scales_i = im_detect_bbox(model, im, box_proposals)
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scales_i


def im_detect_bbox_hflip(model, im, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scales = im_detect_bbox(
        model, im_hf, box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scales


def im_detect_bbox_scale(
    model, im, scale, max_size, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, box_proposals
        )
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(model, im, box_proposals)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
    model, im, aspect_ratio, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model, im_ar, box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(model, im_ar, box_proposals_ar)

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_mask(model, im_scales, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M_HEIGHT = cfg.MRCNN.RESOLUTION_H
    M_WIDTH = cfg.MRCNN.RESOLUTION_W
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scales)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_global_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_global_probs')
    ).squeeze()
    pred_char_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_char_probs')
    ).squeeze()
    # pred_char_boxes = workspace.FetchBlob(
    #     core.ScopedName('mask_fcn_charbox_pred')
    # ).squeeze()
    pred_global_masks = pred_global_masks.reshape([-1, 1, M_HEIGHT, M_WIDTH])
    pred_char_masks = pred_char_masks.reshape([-1, M_HEIGHT, M_WIDTH, 37])
    pred_char_masks = pred_char_masks.transpose([0,3,1,2])
    # pred_char_boxes = pred_char_boxes.reshape([-1, 4,  M_HEIGHT, M_WIDTH])

    return pred_global_masks, pred_char_masks, None


def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    global_masks_ts = []
    char_masks_ts = []
    char_boxes_ts = []

    # Compute masks for the original image (identity transform)
    im_scales_i = im_conv_body_only(model, im)
    global_masks_i, char_masks_i, char_boxes_i = im_detect_mask(model, im_scales_i, boxes)
    global_masks_ts.append(global_masks_i)
    char_masks_ts.append(char_masks_i)
    char_boxes_ts.append(char_boxes_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        global_masks_hf, char_masks_hf, char_boxes_hf = im_detect_mask_hflip(model, im, boxes)
        global_masks_ts.append(global_masks_hf)
        char_masks_ts.append(char_masks_hf)
        char_boxes_ts.append(char_boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        global_masks_scl, char_masks_scl, char_boxes_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        global_masks_ts.append(global_masks_scl)
        char_masks_ts.append(char_masks_scl)
        char_boxes_ts.append(char_boxes_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            global_masks_scl_hf, char_masks_scl_hf, char_boxes_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            global_masks_ts.append(global_masks_scl_hf)
            char_masks_ts.append(char_masks_scl_hf)
            char_boxes_ts.append(char_boxes_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        global_masks_ar, char_masks_ar, char_boxes_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        global_masks_ts.append(global_masks_ar)
        char_masks_ts.append(char_masks_ar)
        char_boxes_ts.append(char_boxes_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            global_masks_ar_hf, char_masks_ar_hf, char_boxes_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            global_masks_ts.append(global_masks_ar_hf)
            char_masks_ts.append(char_masks_ar_hf)
            char_boxes_ts.append(char_boxes_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        global_masks_c = np.mean(global_masks_ts, axis=0)
        char_masks_c = np.mean(char_masks_ts, axis=0)
        # char_boxes_c = np.mean(char_boxes_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        global_masks_c = np.amax(global_masks_ts, axis=0)
        char_masks_c = np.amax(char_masks_ts, axis=0)
        # char_boxes_c = np.amax(char_boxes_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        global_logit_masks = [logit(y) for y in global_masks_ts]
        global_logit_masks = np.mean(global_logit_masks, axis=0)
        global_masks_c = 1.0 / (1.0 + np.exp(-global_logit_masks))

        char_logit_masks = [logit(y) for y in char_masks_ts]
        char_logit_masks = np.mean(char_logit_masks, axis=0)
        char_masks_c = 1.0 / (1.0 + np.exp(-char_logit_masks))

        # char_logit_boxes = [logit(y) for y in char_boxes_ts]
        # char_logit_boxes = np.mean(char_logit_boxes, axis=0)
        # char_boxes_c = 1.0 / (1.0 + np.exp(-char_logit_boxes))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return global_masks_c, char_masks_c, None


def im_detect_mask_hflip(model, im, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    global_masks_hf, char_masks_hf, char_boxes_hf = im_detect_mask(model, im_scales, boxes_hf)

    # Invert the predicted soft masks
    global_masks_inv = global_masks_hf[:, :, :, ::-1]
    # char_masks_inv = char_masks_hf[:, :, :, ::-1]
    # char_boxes_inv = char_boxes_hf[:, :, :, ::-1]

    return global_masks_inv, char_masks_inv, None


def im_detect_mask_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes masks at the given scale."""

    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform mask detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        global_masks_scl, char_masks_scl, char_boxes_scl = im_detect_mask_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        global_masks_scl, char_masks_scl, char_boxes_scl = im_detect_mask(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return global_masks_scl, char_masks_scl, None


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    # boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        global_masks_ar, char_masks_ar, char_boxes_ar = im_detect_mask_hflip(model, im_ar, None)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        global_masks_ar, char_masks_ar, char_boxes_ar = im_detect_mask(model, im_scales, None)

    return global_masks_ar, char_masks_ar, None


def im_detect_keypoints(model, im_scales, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scales)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    im_scales = im_conv_body_only(model, im)
    heatmaps_i = im_detect_keypoints(model, im_scales, boxes)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(model, im, boxes)
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALES[0]
        us_scl = scale > cfg.TEST.SCALES[0]
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_keypoints_hflip(model, im, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    heatmaps_hf = im_detect_keypoints(model, im_scales, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""

    # Store the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        heatmaps_scl = im_detect_keypoints(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False
):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        heatmaps_ar = im_detect_keypoints(model, im_scales, boxes_ar)

    return heatmaps_ar


def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils.boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c


def box_results_with_nms_and_limit(scores, boxes, thresh=0.0001):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M_HEIGHT = cfg.MRCNN.RESOLUTION_H
    M_WIDTH = cfg.MRCNN.RESOLUTION_W
    scale_h = (M_HEIGHT + 2.0) / M_HEIGHT
    scale_w = (M_WIDTH + 2.0) / M_WIDTH
    ref_boxes = box_utils.expand_boxes_hw(ref_boxes, scale_h, scale_w)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M_HEIGHT + 2, M_WIDTH + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            # if cfg.MRCNN.CLS_SPECIFIC_MASK:
            #     padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            # else:
            #     padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]
            padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms



def getstr(seg, charboxes, box_w, box_h, thresh_s=0.15, is_lanms=True, weight_wh=False):
    bg_map = (1 - seg[0, :, :])
    # bg_map = cv2.GaussianBlur(bg_map, (3, 3), sigmaX=3)
    ret, thresh = cv2.threshold(bg_map, 0.15, 1, cv2.THRESH_BINARY)
    # cv2.imwrite('./bin.jpg', (thresh*255).astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
    eroded = cv2.erode(thresh,kernel)
    # cv2.imwrite('./eroded.jpg', (eroded*255).astype(np.uint8))
    # raw_input()
    eroded = eroded.reshape((-1, 1))

    mask_index = np.argmax(seg, axis=0)
    mask_index = mask_index.astype(np.uint8).reshape((-1, 1))
    charboxes = charboxes.transpose([1, 2, 0])  ## 4*h*w -> h*w*4
    ## trans charboxes to x1, y1, x2, y2
    charboxes = dis2xyxy(charboxes, weight_wh)
    scores = seg.transpose([1, 2, 0]).reshape((-1, 37))

    keep_pixels = np.where(eroded ==1)[0]
    mask_index = mask_index[keep_pixels]
    scores = scores[keep_pixels]
    charboxes = charboxes[keep_pixels]

    pos_index = np.where(mask_index > 0)[0]
    mask_index = mask_index[pos_index] ## N*1
    scores = scores[pos_index]  ## N*37
    charboxes = charboxes[pos_index]  ## N*4

    all_charboxes = []
    all_labels = []
    for i in range(1, 37):
        m_idx = np.where(mask_index == i)[0]
        s_idx = np.where(scores[:, i].copy()[m_idx] > thresh_s)[0]
        if s_idx.size >= 1:
            ## nms
            temp_score = scores[:, i].copy()[m_idx][s_idx]
            temp_boxes = charboxes[m_idx][s_idx]
            if is_lanms:
                dets = np.hstack((box2poly(temp_boxes), temp_score[:, np.newaxis])).astype(np.float32, copy=False)
                res_boxes = lanms.merge_quadrangle_n9(dets, 0.3)
                for idx, box in enumerate(res_boxes):
                    mask = np.zeros_like(seg[0, :, :], dtype=np.uint8)
                    box = shrink_single_box(box[:8])
                    cv2.fillPoly(mask, box.reshape((-1, 4, 2)).astype(np.int32), 1)
                    res_boxes[idx, 8] = cv2.mean(seg[i, :, :], mask)[0]

                nms_dets = np.hstack((poly2box(res_boxes[:, :8]), res_boxes[:, -1].reshape((-1, 1))))
            else:
                dets = np.hstack((temp_boxes, temp_score[:, np.newaxis])).astype(np.float32, copy=False)
                keep = box_utils.nms(dets, 0.3)
                nms_dets = dets[keep, :]
            all_charboxes.append(nms_dets)
            all_labels += [i]*(nms_dets.shape[0])
    if len(all_charboxes) > 0:
        all_charboxes = np.vstack(all_charboxes)
        all_labels = np.array(all_labels).reshape((-1, 1))

        ## another nms with high nms thresh to filter out some boxes with high overlap and diferent classes
        keep = box_utils.nms(all_charboxes, 0.6)
        all_labels = all_labels[keep]
        all_charboxes = all_charboxes[keep]
        chars = []
        for i in range(all_charboxes.shape[0]):
            char = {}
            char['x'] = (all_charboxes[i][0] + all_charboxes[i][2])/2.0
            char['y'] = (all_charboxes[i][1] + all_charboxes[i][3])/2.0
            char['s'] = all_charboxes[i][4]
            char['c'] = num2char(all_labels[i])
            char['w'] = (all_charboxes[i][2] - all_charboxes[i][0])
            char['h'] = (all_charboxes[i][3] - all_charboxes[i][1])
            if char['w'] > 3 and char['h'] > 3:
                ## shrink char box
                sx1, sy1, sx2, sy2 = shrink_rect_with_ratio([char['x'], char['y'], char['w'], char['h']], 0.25)
                ## get mean
                cs = seg[1:, sy1:sy2, sx1:sx2].reshape((36, -1)).mean(axis=1).reshape((-1, 1))
                char['cs'] = cs
                chars.append(char)
        chars = sorted(chars, key = lambda x: x['x'])
        string = ''
        score = 0
        scores = []
        css = []
        for char in chars:
            string = string + char['c']
            score += char['s']
            scores.append(char['s'])
            css.append(char['cs'])
        if len(chars) > 0:
            score = score / len(chars)
        return string, score, scores, all_charboxes, np.hstack((css))
    else:
        return '', 0, [], np.zeros((0, 5)), None

def getstr_grid(seg, box_w, box_h):
    pos = 255 - (seg[0]*255).astype(np.uint8)
    mask_index = np.argmax(seg, axis=0)
    mask_index = mask_index.astype(np.uint8)
    pos = pos.astype(np.uint8)
    # seg = seg*255
    # seg = seg.astype(np.uint8)
    ## resize pos and mask_index

    # pos = np.array(Image.fromarray(pos).resize((box_w, box_h)))
    # seg_resize = np.zeros((seg.shape[0], box_h, box_w))
    # for i in range(seg.shape[0]):
    #     seg_resize[i,:,:] = np.array(Image.fromarray(seg[i,:,:]).resize((box_w, box_h)))
    # mask_index = np.array(Image.fromarray(mask_index).resize((box_w, box_h), Image.NEAREST))
    # string, score = seg2text(pos, mask_index, seg_resize)
    string, score, rec_scores = seg2text(pos, mask_index, seg)
    return string, score, rec_scores

def shrink_rect_with_ratio(rect, ratio):
    ## rect:[xc, yc, w, h]
    xc, yc, w, h = rect[0], rect[1], rect[2], rect[3]
    x1, y1, x2, y2 = int(xc - w*ratio), int(yc - h*ratio), int(xc + w*ratio), int(yc + h*ratio)
    ## keep the area of the shrinked box no less than 1 
    if x2 == x1:
        x2 = x1 + 1
    if y2 == y1:
        y2 = y1 + 1
    return x1, y1, x2, y2

def shrink_single_box(poly):
    xc = (poly[0] + poly[2])/2.0
    yc = (poly[1] + poly[7])/2.0
    w_ = (poly[2] - poly[0])/4.0
    h_ = (poly[7] - poly[1])/4.0
    return np.array([xc-w_, yc-h_, xc+w_, yc-h_, xc+w_, yc+h_, xc-w_, yc+h_])





def dis2xyxy(boxes, weight_wh):
    h, w = boxes.shape[0], boxes.shape[1]
    y_index, x_index = np.where(np.ones((h, w)) > 0)
    boxes = boxes.reshape((-1, 4))
    if weight_wh:
        top = (y_index - boxes[:, 0]*h).reshape((-1, 1))
        right = (x_index + boxes[:, 1]*h).reshape((-1, 1))
        bottom = (y_index + boxes[:, 2]*h).reshape((-1, 1))
        left = (x_index - boxes[:, 3]*h).reshape((-1, 1))
    else:
        top = (y_index - boxes[:, 0]*h).reshape((-1, 1))
        right = (x_index + boxes[:, 1]*w).reshape((-1, 1))
        bottom = (y_index + boxes[:, 2]*h).reshape((-1, 1))
        left = (x_index - boxes[:, 3]*w).reshape((-1, 1))
    return np.hstack((left, top, right, bottom))

def box2poly(boxes):
    x1 = boxes[:, 0].copy().reshape((-1, 1))
    y1 = boxes[:, 1].copy().reshape((-1, 1))
    x2 = boxes[:, 2].copy().reshape((-1, 1))
    y2 = boxes[:, 3].copy().reshape((-1, 1))
    return np.hstack((x1, y1, x2, y1, x2, y2, x1, y2))


def poly2box(polys):
    x1 = polys[:, :8:2].min(axis=1).reshape((-1, 1))
    x2 = polys[:, :8:2].max(axis=1).reshape((-1, 1))
    y1 = polys[:, 1:8:2].min(axis=1).reshape((-1, 1))
    y2 = polys[:, 1:8:2].max(axis=1).reshape((-1, 1))
    return np.hstack((x1, y1, x2, y2))


    

# def dis2xyxy(boxes):
#     h, w = boxes.shape[0], boxes.shape[1]
#     y_index, x_index = np.where(np.ones((h, w)) > 0)
#     for i in range(h):
#         for j in range(w):
#             top, right, bottom, left = boxes[i][j][0], boxes[i][j][1], boxes[i][j][2], boxes[i][j][3]
#             boxes[i][j] = np.array([j-left*w, i-top*h, j+right*w, i+bottom*h])
#     return boxes

# def getstr(seg, box_w, box_h):
#     pos = 255 - (seg[0]*255).astype(np.uint8)
#     mask_index = np.argmax(seg, axis=0)
#     mask_index = mask_index.astype(np.uint8)
#     pos = pos.astype(np.uint8)
#     seg = seg*255
#     seg = seg.astype(np.uint8)
#     ## resize pos and mask_index

#     pos = np.array(Image.fromarray(pos).resize((box_w, box_h)))
#     seg_resize = np.zeros((seg.shape[0], box_h, box_w))
#     for i in range(seg.shape[0]):
#         seg_resize[i,:,:] = np.array(Image.fromarray(seg[i,:,:]).resize((box_w, box_h)))
#     mask_index = np.array(Image.fromarray(mask_index).resize((box_w, box_h), Image.NEAREST))
#     string, score = seg2text(pos, mask_index, seg_resize)
#     return string, score

def char2num(char):
    if char in '0123456789':
        num = ord(char) - ord('0') + 1
    elif char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        num = ord(char.lower()) - ord('a') + 11
    else:
        print('error symbol')
        exit()
    return num

def num2char(num):
    if num >=1 and num <=10:
        char = chr(ord('0') + num - 1)
    elif num > 10 and num <= 36:
        char = chr(ord('a') + num - 11)
    else:
        print('error number:%d'%(num))
        exit()
    return char

def seg2text(gray, mask, seg):
    ## input numpy
    img_h, img_w = gray.shape
    ret, thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    scores = []
    for i in range(len(contours)):
        char = {}
        temp = np.zeros((img_h, img_w)).astype(np.uint8)
        cv2.drawContours(temp, [contours[i]], 0, (255), -1)
        x, y, w, h = cv2.boundingRect(contours[i])
        c_x, c_y = x + w/2, y + h/2
        # tmax = 0
        # sym = -1
        # score = 0
        # pixs = mask[temp == 255]
        # seg_contour = seg[:, temp == 255]
        # seg_contour = seg_contour.astype(np.float32) / 255
        # for j in range(1, 37):
        #     tnum = (pixs == j).sum()
        #     if tnum > tmax:
        #         tmax = tnum
        #         sym = j
        # if sym == -1:
        #     continue
        # contour_score = seg_contour[sym,:].sum() / pixs.size

        regions = seg[1:, temp ==255].reshape((36, -1))
        cs = np.mean(regions, axis=1)
        sym = num2char(np.argmax(cs.reshape((-1))) + 1)
        char['x'] = c_x
        char['y'] = c_y
        char['s'] = sym
        char['cs'] = cs.reshape((-1, 1))
        scores.append(np.max(char['cs'], axis=0)[0])

        chars.append(char)
    chars = sorted(chars, key = lambda x: x['x'])
    string = ''
    css = []
    for char in chars:
        string = string + char['s']
        css.append(char['cs'])
    if len(scores)>0:
        score = sum(scores) / len(scores)
    else:
        score = 0.00
    if not css:
        css=[0.]
    return string, score, np.hstack(css)

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points,key = lambda x:x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    if px1<0:
        px1=1
    if px1>image_width:
        px1 = image_width - 1
    if px2<0:
        px2=1
    if px2>image_width:
        px2 = image_width - 1
    if px3<0:
        px3=1
    if px3>image_width:
        px3 = image_width - 1
    if px4<0:
        px4=1
    if px4>image_width:
        px4 = image_width - 1

    if py1<0:
        py1=1
    if py1>image_height:
        py1 = image_height - 1
    if py2<0:
        py2=1
    if py2>image_height:
        py2 = image_height - 1
    if py3<0:
        py3=1
    if py3>image_height:
        py3 = image_height - 1
    if py4<0:
        py4=1
    if py4>image_height:
        py4 = image_height - 1
    return [px1, py1, px2, py2, px3, py3, px4, py4]

def get_polygon(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    # ps = sorted(points,key = lambda x:x[0])
    polygon = []
    for i in range(len(points)):
        point = points[i]
        x = point[0][0] * scale + start_x
        y = point[0][1] * scale + start_y
        polygon.append(x)
        polygon.append(y)
    return polygon

def segm_char_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = 37
    char_strs = [[] for _ in range(cls_boxes[1].shape[0])]
    char_strs_scores = [[] for _ in range(cls_boxes[1].shape[0])]

    M_HEIGHT = cfg.MRCNN.RESOLUTION_H
    M_WIDTH = cfg.MRCNN.RESOLUTION_W
    for k in range(cls_boxes[1].shape[0]):
        text, rec_score = getstr(masks[k,:,:,:], M_HEIGHT, M_WIDTH)
        char_strs.append(text)
        char_strs_scores.append(rec_score)
        # print(text, rec_score)

    return char_strs, char_strs_scores


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (ndarray): array of image scales (relative to im) used
            in the image pyramid
    """
    processed_ims, im_scale_factors = blob_utils.prep_im_for_blob(
        im, cfg.PIXEL_MEANS, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE
    )
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :]**2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if cfg.MODEL.FASTER_RCNN and rois is None:
        height, width = blobs['data'].shape[2], blobs['data'].shape[3]
        scale = im_scale_factors[0]
        blobs['im_info'] = np.array([[height, width, scale]], dtype=np.float32)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors
