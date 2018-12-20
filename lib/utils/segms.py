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

"""Functions for interacting with segmentation masks in the COCO format.

The following terms are used in this module
    mask: a binary mask encoded as a 2D numpy array
    segm: a segmentation mask in one of the two COCO formats (polygon or RLE)
    polygon: COCO's polygon format
    RLE: COCO's run length encoding format
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import pycocotools.mask as mask_util
import cv2


def flip_segms(segms, height, width):
    """Left/right flip each mask in a list of masks."""
    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    def _flip_rle(rle, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            # Magic RLE format handling painfully discovered by looking at the
            # COCO API showAnns function.
            rle = mask_util.frPyObjects([rle], height, width)
        mask = mask_util.decode(rle)
        mask = mask[:, ::-1, :]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    flipped_segms = []
    for segm in segms:
        if type(segm) == list:
            # Polygon format
            flipped_segms.append([_flip_poly(poly, width) for poly in segm])
        else:
            # RLE format
            assert type(segm) == dict
            flipped_segms.append(_flip_rle(segm, height, width))
    return flipped_segms


def polys_to_mask(polygons, height, width):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def mask_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask

def polys_to_mask_wrt_box_rec(rec_rois_gt_chars, polygon, box, M_HEIGHT, M_WIDTH, shrink = 0.5, weight_wh=False):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    char_map = np.zeros((2, M_HEIGHT, M_WIDTH), dtype=np.float32)
    char_weight = np.ones((M_HEIGHT, M_WIDTH), dtype=np.float32)
    char_box = np.zeros((M_HEIGHT, M_WIDTH, 4), dtype=np.float32)
    char_box_inside_weight = np.zeros((M_HEIGHT, M_WIDTH, 4), dtype=np.float32)
    # char_map_weight = np.zeros((2, M_HEIGHT, M_WIDTH), dtype=np.float32)

    xmin = box[0]
    ymin = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygon_norm = np.array(polygon[0], dtype=np.float32)
    polygon_norm[0::2] = (polygon_norm[0::2] - xmin) * M_WIDTH / w
    polygon_norm[1::2] = (polygon_norm[1::2] - ymin) * M_HEIGHT / h
    polygon_reshape = polygon_norm.reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(char_map[0,:,:], [polygon_reshape], 1)

    if rec_rois_gt_chars.size > 0:
        rec_rois_gt_chars[0,:,0:8:2] = (rec_rois_gt_chars[0,:,0:8:2] - xmin) * M_WIDTH / w
        rec_rois_gt_chars[0,:,1:8:2] = (rec_rois_gt_chars[0,:,1:8:2] - ymin) * M_HEIGHT / h
        x_center = np.mean(rec_rois_gt_chars[0,:,0:8:2], axis = 1).astype(np.int32)
        y_center = np.mean(rec_rois_gt_chars[0,:,1:8:2], axis = 1).astype(np.int32)
        for i in range(rec_rois_gt_chars.shape[1]):  
            if x_center[i]>=0 and x_center[i]<M_WIDTH and y_center[i]>=0 and y_center[i]<M_HEIGHT:
                gt_poly = rec_rois_gt_chars[0,i,:8]
                box_xmin = max(0,min(gt_poly[0:8:2]))
                box_xmax = min(M_WIDTH - 1, max(gt_poly[0:8:2]))
                box_ymin = max(0,min(gt_poly[1:8:2]))
                box_ymax = min(M_HEIGHT - 1, max(gt_poly[1:8:2]))
                gt_poly_reshape = gt_poly.reshape((4, 2))
                char_cls = int(rec_rois_gt_chars[0,i,8])
                if shrink>0:
                    # rpoly = _shrink_poly(gt_poly_reshape.copy(), shrink) ## shrink for regression
                    # spoly = _shrink_poly(gt_poly_reshape.copy(), shrink*1.5) ## shrink for classification
                    rpoly = _shrink_rect(gt_poly_reshape.copy(), shrink) ## shrink for regression
                    spoly = _shrink_rect(gt_poly_reshape.copy(), shrink/2) ## shrink for classification
                    # print('gt_poly_reshape', gt_poly_reshape)
                    # print('spoly', spoly)
                else:
                    rpoly = gt_poly_reshape.copy()
                    spoly = gt_poly_reshape.copy()
                rpoly = rpoly.astype(np.int32)
                box_xmin_shrink = max(0, min(rpoly[:,0]))
                box_xmax_shrink = min(M_WIDTH - 1, max(rpoly[:,0]))
                box_ymin_shrink = max(0, min(rpoly[:,1]))
                box_ymax_shrink = min(M_HEIGHT - 1, max(rpoly[:,1]))
            
                if weight_wh:
                    char_box_inside_weight[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 0] = 1.0
                    char_box_inside_weight[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 1] = (box_ymax - box_ymin)*1.0/(box_xmax - box_xmin)
                    char_box_inside_weight[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 2] = 1.0
                    char_box_inside_weight[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 3] = (box_ymax - box_ymin)*1.0/(box_xmax - box_xmin)
                else:
                    char_box_inside_weight[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, :] = 1.0
                tmp_char_box = np.zeros((M_HEIGHT, M_WIDTH))
                tmp_char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink] = 1.0
                index = np.where(tmp_char_box == 1)
                if weight_wh:
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 0] = np.reshape((index[0] - box_ymin) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 1] = np.reshape((box_xmax - index[1]) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 2] = np.reshape((box_ymax - index[0]) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 3] = np.reshape((index[1] - box_xmin) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                else:
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 0] = np.reshape((index[0] - box_ymin) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 1] = np.reshape((box_xmax - index[1]) / float(M_WIDTH), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 2] = np.reshape((box_ymax - index[0]) / float(M_HEIGHT), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                    char_box[box_ymin_shrink:box_ymax_shrink, box_xmin_shrink:box_xmax_shrink, 3] = np.reshape((index[1] - box_xmin) / float(M_WIDTH), (box_ymax_shrink - box_ymin_shrink, box_xmax_shrink - box_xmin_shrink))
                
                ## get classification target
                spoly = spoly.astype(np.int32)
                sbox_xmin_shrink = max(0, min(spoly[:,0]))
                sbox_xmax_shrink = min(M_WIDTH - 1, max(spoly[:,0]))
                sbox_ymin_shrink = max(0, min(spoly[:,1]))
                sbox_ymax_shrink = min(M_HEIGHT - 1, max(spoly[:,1]))

                ## very small char box
                if sbox_xmax_shrink == sbox_xmin_shrink:
                    sbox_xmax_shrink = sbox_xmin_shrink + 1
                if sbox_ymax_shrink == sbox_ymin_shrink:
                    sbox_ymax_shrink = sbox_ymin_shrink + 1

                char_map[1, sbox_ymin_shrink:sbox_ymax_shrink, sbox_xmin_shrink:sbox_xmax_shrink] = char_cls

        ## char_weight 
        pos_index = np.where(char_map[1, :, :] > 0)
        pos_num = pos_index[0].size
        if pos_num > 0:
            pos_weight = 1.0 * (M_WIDTH*M_HEIGHT - pos_num)/pos_num
            char_weight[pos_index] = pos_weight
    else: ## for samples without char ann
        char_map[1, :, :].fill(-1)


    return char_map, char_weight, char_box, char_box_inside_weight


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def rle_mask_voting(
    top_masks, all_masks, all_dets, iou_thresh, binarize_thresh, method='AVG'
):
    """Returns new masks (in correspondence with `top_masks`) by combining
    multiple overlapping masks coming from the pool of `all_masks`. Two methods
    for combining masks are supported: 'AVG' uses a weighted average of
    overlapping mask pixels; 'UNION' takes the union of all mask pixels.
    """
    if len(top_masks) == 0:
        return

    all_not_crowd = [False] * len(all_masks)
    top_to_all_overlaps = mask_util.iou(top_masks, all_masks, all_not_crowd)
    decoded_all_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in all_masks
    ]
    decoded_top_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in top_masks
    ]
    all_boxes = all_dets[:, :4].astype(np.int32)
    all_scores = all_dets[:, 4]

    # Fill box support with weights
    mask_shape = decoded_all_masks[0].shape
    mask_weights = np.zeros((len(all_masks), mask_shape[0], mask_shape[1]))
    for k in range(len(all_masks)):
        ref_box = all_boxes[k]
        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, mask_shape[1])
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, mask_shape[0])
        mask_weights[k, y_0:y_1, x_0:x_1] = all_scores[k]
    mask_weights = np.maximum(mask_weights, 1e-5)

    top_segms_out = []
    for k in range(len(top_masks)):
        # Corner case of empty mask
        if decoded_top_masks[k].sum() == 0:
            top_segms_out.append(top_masks[k])
            continue

        inds_to_vote = np.where(top_to_all_overlaps[k] >= iou_thresh)[0]
        # Only matches itself
        if len(inds_to_vote) == 1:
            top_segms_out.append(top_masks[k])
            continue

        masks_to_vote = [decoded_all_masks[i] for i in inds_to_vote]
        if method == 'AVG':
            ws = mask_weights[inds_to_vote]
            soft_mask = np.average(masks_to_vote, axis=0, weights=ws)
            mask = np.array(soft_mask > binarize_thresh, dtype=np.uint8)
        elif method == 'UNION':
            # Any pixel that's on joins the mask
            soft_mask = np.sum(masks_to_vote, axis=0)
            mask = np.array(soft_mask > 1e-5, dtype=np.uint8)
        else:
            raise NotImplementedError('Method {} is unknown'.format(method))
        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        top_segms_out.append(rle)

    return top_segms_out


def rle_mask_nms(masks, dets, thresh, mode='IOU'):
    """Performs greedy non-maximum suppression based on an overlap measurement
    between masks. The type of measurement is determined by `mode` and can be
    either 'IOU' (standard intersection over union) or 'IOMA' (intersection over
    mininum area).
    """
    if len(masks) == 0:
        return []
    if len(masks) == 1:
        return [0]

    if mode == 'IOU':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(union(m1, m2))
        all_not_crowds = [False] * len(masks)
        ious = mask_util.iou(masks, masks, all_not_crowds)
    elif mode == 'IOMA':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / min(area(m1), area(m2))
        all_crowds = [True] * len(masks)
        # ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
        ious = mask_util.iou(masks, masks, all_crowds)
        # ... = max(area(intersect(m1, m2)) / area(m2),
        #           area(intersect(m2, m1)) / area(m1))
        ious = np.maximum(ious, ious.transpose())
    elif mode == 'CONTAINMENT':
        # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
        # Which measures how much m2 is contained inside m1
        all_crowds = [True] * len(masks)
        ious = mask_util.iou(masks, masks, all_crowds)
    else:
        raise NotImplementedError('Mode {} is unknown'.format(mode))

    scores = dets[:, 4]
    order = np.argsort(-scores)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = ious[i, order[1:]]
        inds_to_keep = np.where(ovr <= thresh)[0]
        order = order[inds_to_keep + 1]

    return keep


def rle_masks_to_boxes(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""
    if len(masks) == 0:
        return []

    decoded_masks = [
        np.array(mask_util.decode(rle), dtype=np.float32) for rle in masks
    ]

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()

    boxes = np.zeros((len(decoded_masks), 4))
    keep = [True] * len(decoded_masks)
    for i, mask in enumerate(decoded_masks):
        if mask.sum() == 0:
            keep[i] = False
            continue
        flat_mask = mask.sum(axis=0)
        x0, x1 = get_bounds(flat_mask)
        flat_mask = mask.sum(axis=1)
        y0, y1 = get_bounds(flat_mask)
        boxes[i, :] = (x0, y0, x1, y1)

    return boxes, np.where(keep)[0]

def _shrink_rect(poly, shrink):
    xmin = min(poly[:,0])
    xmax = max(poly[:,0])
    ymin = min(poly[:,1])
    ymax = max(poly[:,1])
    assert xmax > xmin and ymax > ymin
    xc = (xmax + xmin) / 2
    yc = (ymax + ymin) / 2
    w = xmax - xmin
    h = ymax - ymin
    sxmin = xc - w/2*shrink
    sxmax = xc + w/2*shrink
    symin = yc - h/2*shrink
    symax = yc + h/2*shrink
    return np.array([sxmin, symin, sxmax, symin, sxmax, symax, sxmin, symax]).reshape((4, 2))

def _shrink_poly(poly, shrink):
    # shrink ratio
    R = shrink
    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                   np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly
