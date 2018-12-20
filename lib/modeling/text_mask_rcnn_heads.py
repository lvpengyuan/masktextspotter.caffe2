# Modified by Minghui Liao and Pengyuan Lyu
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

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
import modeling.ResNet as ResNet
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_mask_rcnn_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = 37


    # Predict mask using Conv
    # Use GaussianFill for class-agnostic mask prediction; fills based on
    # fan-in can be too large in this case and cause divergence
    fill = (
        cfg.MRCNN.CONV_INIT
        if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
    )
    blob_out_global = model.Conv(
        blob_in,
        'mask_fcn_global_logits',
        dim,
        1,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(fill, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )

    blob_out_char = model.Conv(
        blob_in,
        'mask_fcn_char_logits',
        dim,
        num_cls,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(fill, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )

    # blob_out_charbox_pred = model.Conv(
    #     blob_in,
    #     'mask_fcn_charbox_pred',
    #     dim,
    #     4,
    #     kernel=1,
    #     pad=0,
    #     stride=1,
    #     weight_init=(fill, {'std': 0.001}),
    #     bias_init=const_fill(0.0)
    # )

    if cfg.MRCNN.UPSAMPLE_RATIO > 1:
        blob_out_global = model.BilinearInterpolation(
            'mask_fcn_global_logits', 'mask_fcn_global_logits_up', 1, 1,
            cfg.MRCNN.UPSAMPLE_RATIO
        )
        blob_out_char = model.BilinearInterpolation(
            'mask_fcn_char_logits', 'mask_fcn_char_logits_up', num_cls, num_cls,
            cfg.MRCNN.UPSAMPLE_RATIO
        )
        # blob_out_charbox_pred = model.BilinearInterpolation(
        #     'mask_fcn_charbox_pred', 'mask_fcn_charbox_pred_up', 4, 4,
        #     cfg.MRCNN.UPSAMPLE_RATIO
        # )

    ## transpose and reshape box_pred
    # if model.train:
        # blob_out_charbox_pred = model.net.Transpose(blob_out_charbox_pred, 'blob_out_charbox_pred_trans', axes=[0,2,3,1])
        # blob_out_charbox_pred, _ = model.net.Reshape(blob_out_charbox_pred, ['blob_out_charbox_pred_reshape', 'blob_out_charbox_pred_old_shape'], shape=(-1, 4))
    
    if not model.train:  # == if test
        blob_out_global = model.net.Sigmoid(blob_out_global, 'mask_fcn_global_probs')
        blob_out_char = model.net.Transpose(blob_out_char, 'blob_out_char_trans', axes=[0,2,3,1])
        blob_out_char, _ = model.net.Reshape(blob_out_char, ['blob_out_char_reshape', 'blob_out_char_old_shape'], shape=(-1, 37))
        blob_out_char = model.net.Softmax(blob_out_char, 'mask_fcn_char_probs', axis=1)


    return [blob_out_global, blob_out_char]


def add_mask_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    loss_global_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask[0], 'masks_global_int32'],
        'loss_global_mask',
        scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    mask_cls_prob, loss_char_mask = model.net.SpatialSoftmaxWithLoss(
        [blob_mask[1], 'masks_char_int32', 'masks_char_weight'],
        ['mask_cls_prob', 'loss_char_mask'],
        scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    # loss_char_bbox = model.net.SmoothL1Loss(
    #     [
    #         blob_mask[2], 'char_bbox_targets', 'char_bbox_inside_weights',
    #         'char_bbox_outside_weights'
    #     ],
    #     'loss_char_bbox',
    #     scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_CHAR_BOX
    # )

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_global_mask, loss_char_mask])
    model.AddLosses(['loss_global_mask', 'loss_char_mask'])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

def mask_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )


def mask_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale,
        resolution_w=cfg.MRCNN.ROI_XFORM_RESOLUTION_W,
        resolution_h=cfg.MRCNN.ROI_XFORM_RESOLUTION_H,
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_mask_int32'],
            ['_[mask]_res5_2_sum_sliced']
        )
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
            model,
            blob_in,
            dim_in,
            spatial_scale
        )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=const_fill(0.0)
    )
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def mask_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
        model,
        blob_in,
        dim_in,
        spatial_scale
    )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_masks(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale,
        resolution_w=cfg.MRCNN.ROI_XFORM_RESOLUTION_W,
        resolution_h=cfg.MRCNN.ROI_XFORM_RESOLUTION_H,
    )

    dilation = cfg.MRCNN.DILATION
    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[mask]_res5',
        '_[mask]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init
    )

    return s, 2048
