# Modified by Pengyuan Lyu
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

"""Detectron data loader. The design is generic and abstracted away from any
details of the minibatch. A minibatch is a dictionary of blob name keys and
their associated numpy (float32 or int32) ndarray values.

Outline of the data loader design:

loader thread\
loader thread \                    / GPU 1 enqueue thread -> feed -> EnqueueOp
...           -> minibatch queue ->  ...
loader thread /                    \ GPU N enqueue thread -> feed -> EnqueueOp
loader thread/

<---------------------------- CPU -----------------------------|---- GPU ---->

A pool of loader threads construct minibatches that are put onto the shared
minibatch queue. Each GPU has an enqueue thread that pulls a minibatch off the
minibatch queue, feeds the minibatch blobs into the workspace, and then runs
an EnqueueBlobsOp to place the minibatch blobs into the GPU's blobs queue.
During each fprop the first thing the network does is run a DequeueBlobsOp
in order to populate the workspace with the blobs from a queued minibatch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from collections import OrderedDict
import logging
import numpy as np
import Queue
import signal
import threading
import time
import uuid
import random

from caffe2.python import core, workspace

from core.config import cfg
from roi_data.minibatch import get_minibatch
from roi_data.minibatch import get_minibatch_blob_names
from utils.coordinator import coordinated_get
from utils.coordinator import coordinated_put
from utils.coordinator import Coordinator
import utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class MixRoIDataLoader(object):
    def __init__(
        self,
        roidbs,
        num_loaders=4,
        minibatch_queue_size=64,
        blobs_queue_capacity=8
    ):
        self._roidbs = roidbs
        self._lock = threading.Lock()
        self._perms = [deque(range(len(roidb))) for roidb in self._roidbs]
        self._curs = [0 for roidb in self._roidbs]  # _perm cursor
        self._num_ratios = cfg.TRAIN.MIX_RATIOS
        # The minibatch queue holds prepared training data in host (CPU) memory
        # When training with N > 1 GPUs, each element in the minibatch queue
        # is actually a partial minibatch which contributes 1 / N of the
        # examples to the overall minibatch
        self._minibatch_queue = Queue.Queue(maxsize=minibatch_queue_size)
        self._blobs_queue_capacity = blobs_queue_capacity
        # Random queue name in case one instantiates multple RoIDataLoaders
        self._loader_id = uuid.uuid4()
        self._blobs_queue_name = 'roi_blobs_queue_{}'.format(self._loader_id)
        # Loader threads construct (partial) minibatches and put them on the
        # minibatch queue
        self._num_loaders = num_loaders
        self._num_gpus = cfg.NUM_GPUS
        self.coordinator = Coordinator()

        self._output_names = get_minibatch_blob_names()
        # self._shuffle_roidb_inds()
        self._shuffle_roidb_inds_init()
        self._get_dataset_inds()
        self.create_threads()


    def minibatch_loader_thread(self):
        """Load mini-batches and put them onto the mini-batch queue."""
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                blobs = self.get_next_minibatch()
                # Blobs must be queued in the order specified by
                # self.get_output_names
                ordered_blobs = OrderedDict()
                for key in self.get_output_names():
                    assert blobs[key].dtype in (np.int32, np.float32), \
                        'Blob {} of dtype {} must have dtype of ' \
                        'np.int32 or np.float32'.format(key, blobs[key].dtype)
                    ordered_blobs[key] = blobs[key]
                coordinated_put(
                    self.coordinator, self._minibatch_queue, ordered_blobs
                )
        logger.info('Stopping mini-batch loading thread')

    def enqueue_blobs_thread(self, gpu_id, blob_names):
        """Transfer mini-batches from a mini-batch queue to a BlobsQueue."""
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                if self._minibatch_queue.qsize == 0:
                    logger.warning('Mini-batch queue is empty')
                blobs = coordinated_get(self.coordinator, self._minibatch_queue)
                self.enqueue_blobs(gpu_id, blob_names, blobs.values())
                logger.debug(
                    'batch queue size {}'.format(self._minibatch_queue.qsize())
                )
            logger.info('Stopping enqueue thread')

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch. Thread safe."""
        valid = False
        while not valid:
            db_inds, dataset_ind = self._get_next_minibatch_inds()
            minibatch_db = [self._roidbs[dataset_ind][i] for i in db_inds]
            blobs, valid = get_minibatch(minibatch_db)
            # for index, i in enumerate(db_inds):
            #     self._roidbs[dataset_ind][i] = new_roidb[index]
        return blobs

    def _get_dataset_inds(self):
        self._dataset_inds = []
        nums = [round(x*self._num_gpus*cfg.TRAIN.IMS_PER_BATCH) for x in self._num_ratios]
        for i in range(len(self._num_ratios)):
            self._dataset_inds += [i]*int(nums[i])
        assert(len(self._dataset_inds) == self._num_gpus*cfg.TRAIN.IMS_PER_BATCH)
        random.shuffle(self._dataset_inds)


    def _shuffle_roidb_inds_init(self):
        """Randomly permute the training roidb. Not thread safe."""
        if cfg.TRAIN.MIX_TRAIN:
           assert cfg.TRAIN.ASPECT_GROUPING == False, 'ASPECT_GROUPING should be set to false, for we random select roidbs from different sets'
        
        self._perms = [np.random.permutation(np.arange(len(roidb))) for roidb in self._roidbs]
        self._perms = [deque(perm) for perm in self._perms]
        self._curs = [0 for roidb in self._roidbs]

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb. Not thread safe."""
        if cfg.TRAIN.MIX_TRAIN:
           assert cfg.TRAIN.ASPECT_GROUPING == False, 'ASPECT_GROUPING should be set to false, for we random select roidbs from different sets'
        
        for i, cur in enumerate(self._curs):
            if cur >= len(self._perms[i]):
                self._perms[i] = deque(np.random.permutation(np.arange(len(self._roidbs[i]))))
                self._curs[i] = 0




    # def _get_next_minibatch_inds(self):
    #     """Return the roidb indices for the next minibatch. Thread safe."""
    #     with self._lock:
    #         # We use a deque and always take the *first* IMS_PER_BATCH items
    #         # followed by *rotating* the deque so that we see fresh items
    #         # each time. If the length of _perm is not divisible by
    #         # IMS_PER_BATCH, then we end up wrapping around the permutation.
    #         db_inds = [self._perm[i] for i in range(cfg.TRAIN.IMS_PER_BATCH)]
    #         self._perm.rotate(-cfg.TRAIN.IMS_PER_BATCH)
    #         self._cur += cfg.TRAIN.IMS_PER_BATCH
    #         if self._cur >= len(self._perm):
    #             self._shuffle_roidb_inds()
    #     return db_inds

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch. Thread safe."""
        with self._lock:
            # We use a deque and always take the *first* IMS_PER_BATCH items
            # followed by *rotating* the deque so that we see fresh items
            # each time. If the length of _perm is not divisible by
            # IMS_PER_BATCH, then we end up wrapping around the permutation.
            ##select roidb from different datasets

            dataset_ind = random.choice(self._dataset_inds)
            db_inds = [self._perms[dataset_ind][i] for i in range(cfg.TRAIN.IMS_PER_BATCH)]
            self._perms[dataset_ind].rotate(-cfg.TRAIN.IMS_PER_BATCH)
            self._curs[dataset_ind] += cfg.TRAIN.IMS_PER_BATCH
            if self._curs[dataset_ind] >= len(self._perms[dataset_ind]):
                self._shuffle_roidb_inds()
        return db_inds, dataset_ind

    def get_output_names(self):
        return self._output_names

    def enqueue_blobs(self, gpu_id, blob_names, blobs):
        """Put a mini-batch on a BlobsQueue."""
        assert len(blob_names) == len(blobs)
        t = time.time()
        dev = c2_utils.CudaDevice(gpu_id)
        queue_name = 'gpu_{}/{}'.format(gpu_id, self._blobs_queue_name)
        blob_names = ['gpu_{}/{}'.format(gpu_id, b) for b in blob_names]
        for (blob_name, blob) in zip(blob_names, blobs):
            workspace.FeedBlob(blob_name, blob, device_option=dev)
        logger.debug(
            'enqueue_blobs {}: workspace.FeedBlob: {}'.
            format(gpu_id, time.time() - t)
        )
        t = time.time()
        op = core.CreateOperator(
            'SafeEnqueueBlobs', [queue_name] + blob_names,
            blob_names + [queue_name + '_enqueue_status'],
            device_option=dev
        )
        workspace.RunOperatorOnce(op)
        logger.debug(
            'enqueue_blobs {}: workspace.RunOperatorOnce: {}'.
            format(gpu_id, time.time() - t)
        )

    def create_threads(self):
        # Create mini-batch loader threads, each of which builds mini-batches
        # and places them into a queue in CPU memory
        self._workers = [
            threading.Thread(target=self.minibatch_loader_thread)
            for _ in range(self._num_loaders)
        ]

        # Create one BlobsQueue per GPU
        # (enqueue_blob_names are unscoped)
        enqueue_blob_names = self.create_blobs_queues()

        # Create one enqueuer thread per GPU
        self._enqueuers = [
            threading.Thread(
                target=self.enqueue_blobs_thread,
                args=(gpu_id, enqueue_blob_names)
            ) for gpu_id in range(self._num_gpus)
        ]

    def start(self, prefill=False):
        for w in self._workers + self._enqueuers:
            w.start()
        if prefill:
            logger.info('Pre-filling mini-batch queue...')
            while not self._minibatch_queue.full():
                logger.info(
                    '  [{:d}/{:d}]'.format(
                        self._minibatch_queue.qsize(),
                        self._minibatch_queue.maxsize
                    )
                )
                time.sleep(0.1)
                # Detect failure and shutdown
                if self.coordinator.should_stop():
                    self.shutdown()
                    break

    def shutdown(self):
        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        self.close_blobs_queues()
        for w in self._workers + self._enqueuers:
            w.join()

    def create_blobs_queues(self):
        """Create one BlobsQueue for each GPU to hold mini-batches."""
        for gpu_id in range(self._num_gpus):
            with c2_utils.GpuNameScope(gpu_id):
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        'CreateBlobsQueue', [], [self._blobs_queue_name],
                        num_blobs=len(self.get_output_names()),
                        capacity=self._blobs_queue_capacity
                    )
                )
        return self.create_enqueue_blobs()

    def close_blobs_queues(self):
        """Close a BlobsQueue."""
        for gpu_id in range(self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        'CloseBlobsQueue', [self._blobs_queue_name], []
                    )
                )

    def create_enqueue_blobs(self):
        blob_names = self.get_output_names()
        enqueue_blob_names = [
            '{}_enqueue_{}'.format(b, self._loader_id) for b in blob_names
        ]
        for gpu_id in range(self._num_gpus):
            with c2_utils.NamedCudaScope(gpu_id):
                for blob in enqueue_blob_names:
                    workspace.CreateBlob(core.ScopedName(blob))
        return enqueue_blob_names

    def register_sigint_handler(self):
        def signal_handler(signal, frame):
            logger.info(
                'SIGINT: Shutting down RoIDataLoader threads and exiting...'
            )
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
