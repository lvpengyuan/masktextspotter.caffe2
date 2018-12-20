# Modified by Pengyuan Lyu
# ##############################################################################
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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_CACHE_DIR= os.path.join(os.path.dirname(__file__), 'cache')
if not os.path.exists(_CACHE_DIR):
    os.mkdir(_CACHE_DIR)
IM_DIR = 'im_dir'
ANN_FN = 'ann_fn'
IM_LIST='im_list'
IM_PREFIX = 'image_prefix'

# Available datasets
DATASETS = {
    'synth_train': {
        IM_DIR:
            _DATA_DIR + '/synth/train_images',
        ANN_FN:
            _DATA_DIR + '/synth/train_gts',
        IM_LIST:
            _DATA_DIR + '/synth/train_list.txt'
    },
    'synth0_train': {
        IM_DIR:
            _DATA_DIR + '/synth0/train_images',
        ANN_FN:
            _DATA_DIR + '/synth0/train_gts',
        IM_LIST:
            _DATA_DIR + '/synth0/train_list.txt'
    },
    'icdar2013_train': {
        IM_DIR:
            _DATA_DIR + '/icdar2013/train_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/icdar2013/train_gts',
        IM_LIST:
            _DATA_DIR + '/icdar2013/train_list.txt'
    },
    'icdar2013_test': {
        IM_DIR:
            _DATA_DIR + '/icdar2013/test_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/icdar2013/test_gts',
        IM_LIST:
            _DATA_DIR + '/icdar2013/test_list.txt',
    },
    'icdar2015_train': {
        IM_DIR:
            _DATA_DIR + '/icdar2015/train_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/icdar2015/train_gts',
        IM_LIST:
            _DATA_DIR + '/icdar2015/train_list.txt'
    },
    'icdar2015_test': {
        IM_DIR:
            _DATA_DIR + '/icdar2015/test_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/icdar2015/test_gts',
        IM_LIST:
            _DATA_DIR + '/icdar2015/test_list.txt'
    },
    'totaltext_train': {
        IM_DIR:
            _DATA_DIR + '/totaltext/train_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/totaltext/train_gts',
        IM_LIST:
            _DATA_DIR + '/totaltext/train_list.txt'
    },
    'totaltext_test': {
        IM_DIR:
            _DATA_DIR + '/totaltext/test_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/totaltext/test_gts',
        IM_LIST:
            _DATA_DIR + '/totaltext/test_list.txt'
    },
    'scut-eng-char_train': {
        IM_DIR:
            _DATA_DIR + '/scut-eng-char/train_images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/scut-eng-char/train_gts',
        IM_LIST:
            _DATA_DIR + '/scut-eng-char/train_list.txt'
    }
}