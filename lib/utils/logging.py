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

"""Utilities for logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from email.mime.text import MIMEText
import json
import logging
import numpy as np
import smtplib
import sys
from datetime import datetime
import time
# Print lower precision floating point values than default FLOAT_REPR
json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')


def log_json_stats(stats, sort_keys=True):
    # print('{:s} json_stats: {:s}'.format(str(datetime.now()), json.dumps(stats, sort_keys=sort_keys)))
    # print('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))
    logging.info('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count


def send_email(subject, body, to):
    s = smtplib.SMTP('localhost')
    mime = MIMEText(body)
    mime['Subject'] = subject
    mime['To'] = to
    s.sendmail('detectron', to, mime.as_string())

def setup_logger(log_file_path):
    """
    adapted from https://github.com/lvpengyuan/ssd.tf/blob/fctd-box/src/utils.py
    Setup a logger that simultaneously output to a file and stdout
    ARGS
    log_file_path: string, path to the logging file
    """
    # logging settings
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    # file handler
    log_file_handler = logging.FileHandler(log_file_path + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    # stream handler (stdout)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_file_path)


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger
