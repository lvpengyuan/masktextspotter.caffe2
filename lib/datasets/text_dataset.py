#-*-coding:utf-8-*- 
# Created by Pengyuan Lyu
#####################################################################

""" We create a new class for text datasets for convenience. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
from PIL import Image
import cPickle
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils

from datasets.textdataset_catalog import DATASETS
from datasets.textdataset_catalog import ANN_FN
from datasets.textdataset_catalog import IM_DIR
from datasets.textdataset_catalog import IM_LIST
from datasets.textdataset_catalog import _CACHE_DIR

logger = logging.getLogger(__name__)
DEBUG = False


class TextDataSet(object):
    """ A class representing a text dataset """
    def __init__(self, name):
        assert name in DATASETS.keys(), 'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), 'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        # assert os.path.exists(DATASETS[name][ANN_FN]), 'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        assert os.path.exists(DATASETS[name][IM_LIST]), 'Annotation file \'{}\' not found'.format(DATASETS[name][IM_LIST])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_name = name.strip().split('_')[0]
        self.set = name.strip().split('_')[-1]
        self.im_list = DATASETS[name][IM_LIST]
        self.image_directory = DATASETS[name][IM_DIR]
        self.ann_directory = DATASETS[name][ANN_FN]
        self.debug_timer = Timer()
        self.classes = ['__background__', 'text']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index() 
        self.num_images = len(self.image_set_index)
        self.char_classes = '_0123456789abcdefghijklmnopqrstuvwxyz'

    def get_roidb(self, gt=None, proposal_file=None, use_charann=True, min_proposal_size=2, proposal_limit=-1, crowd_filter_thresh=0):
        """Return an roidb corresponding to the dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """

        assert proposal_file is None, 'Only gt_roidb is supported Now!!!'
        self.min_proposal_size = min_proposal_size
        self.keypoints = None
        self.use_charann = use_charann
        self.debug_timer.tic()
        roidb = self.gt_roidb()
        logger.debug(
            '_add_gt_annotations took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
        )

        return roidb

    def load_image_set_index(self):
        image_set_index_file = os.path.join(self.im_list)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        image_file = os.path.join(self.image_directory, index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        if self.set == 'train':
            if self.use_charann == False:
                cache_file = os.path.join(_CACHE_DIR, self.name + '_gt_roidb_wocharann.pkl')
            else:
                cache_file = os.path.join(_CACHE_DIR, self.name + '_gt_roidb.pkl')
        else:
            cache_file = os.path.join(_CACHE_DIR, self.name + '_val_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            return roidb
        gt_roidb = []
        for i, index in enumerate(self.image_set_index):
            if i % 1000 == 0:
                print(i, len(self.image_set_index))
            gt_roidb.append(self.load_text_annotation(index))
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        return gt_roidb


    def line2boxes(self, line):
        if self.name == 'totaltext_train':
            parts = line.strip().split(',')
            return [parts[-1]], np.array([[float(x) for x in parts[:-1]]])
        else:
            parts = line.strip().split(',')
            if '\xef\xbb\xbf' in parts[0]:
                parts[0] = parts[0][3:]
            if '\ufeff' in parts[0]:
                parts[0] = parts[0].replace('\ufeff', '')
            x1 = np.array([int(float(x)) for x in parts[::9]])
            y1 = np.array([int(float(x)) for x in parts[1::9]])
            x2 = np.array([int(float(x)) for x in parts[2::9]])
            y2 = np.array([int(float(x)) for x in parts[3::9]])
            x3 = np.array([int(float(x)) for x in parts[4::9]])
            y3 = np.array([int(float(x)) for x in parts[5::9]])
            x4 = np.array([int(float(x)) for x in parts[6::9]])
            y4 = np.array([int(float(x)) for x in parts[7::9]])
            strs = parts[8::9]
            loc = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
            return strs, loc

    def char2num(self, chars):
        ## chars ['h', 'e', 'l', 'l', 'o']
        nums = [self.char_classes.index(c.lower()) for c in chars]
        return nums

    def check_charbbs(self, charbbs):
        xmins = np.minimum.reduce([charbbs[:,0], charbbs[:,2], charbbs[:,4], charbbs[:,6]])
        xmaxs = np.maximum.reduce([charbbs[:,0], charbbs[:,2], charbbs[:,4], charbbs[:,6]])
        ymins = np.minimum.reduce([charbbs[:,1], charbbs[:,3], charbbs[:,5], charbbs[:,7]])
        ymaxs = np.maximum.reduce([charbbs[:,1], charbbs[:,3], charbbs[:,5], charbbs[:,7]])
        return np.logical_and(xmaxs - xmins > self.min_proposal_size, ymaxs - ymins > self.min_proposal_size)

    def load_gt_from_txt(self, gt_path, height, width):
        lines = open(gt_path).readlines()
        words, boxes, polygons, charboxes, seg_areas, segmentations = [], [], [], [], [], []
        for line in  lines:
            strs, loc = self.line2boxes(line)
            word = strs[0]
            if word == '###':
                continue
            else:
                rect = list(loc[0])
                min_x = min(rect[::2]) - 1
                min_y = min(rect[1::2]) - 1
                max_x = max(rect[::2]) - 1
                max_y = max(rect[1::2]) - 1
                box = [min_x, min_y, max_x, max_y]
                area = (max_x - min_x) * (max_y - min_y)
    
                ## filter out small objects and assign an index for the kept box
                if max_x - min_x  <= self.min_proposal_size  or max_y - min_y <= self.min_proposal_size or min_x < 0 or min_y < 0 or max_x>width or max_y>height or area < cfg.TRAIN.GT_MIN_AREA:
                    continue
                else:
                    tindex = len(boxes)
                    boxes.append(box)
                    seg_areas.append(area)
                    if self.name == 'totaltext_train':
                        polygons.append(np.array((list(loc[0])*4)[:8]))
                        segmentations.append([list(loc[0])])
                    else:
                        polygons.append(loc[0, :])
                        segmentations.append([[loc[0][0], loc[0][1], loc[0][2], loc[0][3], loc[0][4], loc[0][5], loc[0][6], loc[0][7]]])
                    words.append(strs)
                    if loc.shape[0] == 1 :
                        charbbs = np.zeros((0, 10), dtype=np.float32)
                    else:
                        charbbs = np.zeros((loc.shape[0] - 1, 10), dtype=np.float32)
                    ## char2num
                    c_class = self.char2num(strs[1:])
                    if loc.shape[0] > 1:
                        charbbs[:, :8] = loc[1:, :]
                        valid = self.check_charbbs(charbbs)
                        # print(valid)
                        charbbs = charbbs[valid]
                        charbbs[:, 8] = np.array(c_class)[valid]
                        charbbs[:, 9] = tindex
                        if charbbs.shape[0] > 0:
                            charboxes.append(charbbs)
        if len(boxes) > 0:
            if self.use_charann:
                return words, np.array(boxes), np.array(polygons), np.vstack(charboxes), np.array(seg_areas), segmentations
            else:
                charbbs = np.zeros((0, 10), dtype=np.float32)
                return words, np.array(boxes), np.array(polygons), charbbs, np.array(seg_areas), segmentations
        else:
            return [], np.zeros((0, 4), dtype=np.float32), np.zeros((0, 8), dtype=np.float32), np.zeros((0, 10), dtype=np.float32), np.zeros((0), dtype=np.float32), []

    def load_text_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from gt file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        # import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['dataset'] = self
        roi_rec['has_visible_keypoints'] = False
        roi_rec['image'] = self.image_path_from_index(index)
        filename = os.path.join(self.ann_directory, index + '.txt')
        size = Image.open(roi_rec['image']).size
        roi_rec['height'] = size[1]
        roi_rec['width'] = size[0]

        if self.set == 'train':

            ## get objs
            #keep_boxes: rectangle boxes, n*4
            #keep_polygons: polygons boxes, n*8 corresponding to keep_boxes
            #keep_charboxes: charboxes, n*10 corresponding to keep_boxes dim0-7 for loc information, dim8 for charbox class, dim9 to recoder the corresponding word box
            words, keep_boxes, keep_polygons, keep_charboxes, seg_areas, segmentations = self.load_gt_from_txt(filename, size[1], size[0])
            if DEBUG:
                print('words', words)
                print('keep_boxes', keep_boxes)
                print('keep_polygons', keep_polygons)
                print('keep_charboxes', keep_charboxes)
                print('seg_areas', seg_areas)
                print('segmentations', segmentations)


            num_objs = keep_boxes.shape[0]
            ## Note that, we use a flag(0/1) to decide whether a box can be traind end2end.
            boxes = np.zeros((num_objs, 4), dtype=np.float32)
            gt_classes = np.ones((num_objs), dtype=np.int32) ## we only have text class, so gt will be always 1.
            
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps.fill(-1)
            box_to_gt_ind_map = np.array(range(num_objs))
            iscrowd = np.zeros((num_objs), dtype=bool)
            is_e2e = False
            overlaps[:, 1] = 1
            boxes[:, :4] = keep_boxes
            # if self.image_name != 'icdar2015':-
            #     boxes[:, 4] = 1
            #     is_e2e = True

            roi_rec.update({'boxes': boxes,                   ## np n*5
                            'polygons': keep_polygons.astype(np.float32),        ## np n*8
                            'charboxes': keep_charboxes.astype(np.float32),      ## np k*10
                            'words': words,                   ## [[hello, h, e, l, l, o], ...]
                            'gt_classes': gt_classes,         ## np n
                            'gt_overlaps': scipy.sparse.csr_matrix(overlaps),          ## np n*2
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'flipped': False,
                            'is_e2e': is_e2e,
                            'seg_areas': seg_areas.astype(np.float32),
                            'box_to_gt_ind_map': box_to_gt_ind_map.astype(np.int32),
                            'segms': segmentations,
                            'is_crowd': iscrowd
                            })
            return roi_rec
        else:
            return roi_rec




    