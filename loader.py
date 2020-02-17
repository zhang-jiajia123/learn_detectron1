from collections import deque
from collections import OrderedDict
import logging
import numpy as np
import signal
#import threading
import time
import uuid
from six.moves import queue as Queue
import minibatch


IMS_PER_BATCH = 2

class RoIDataLoader(object):
    def __init__(
        self,
        roidb,
        num_loaders=4,
        minibatch_queue_size=64,
        blobs_queue_capacity=8
    ):
        self._roidb = roidb
        #self._lock = threading.Lock()
        self._perm = deque(range(len(self._roidb)))
        self._cur = 0  # _perm cursor
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
        self._num_gpus = 1
        #self.coordinator = Coordinator()

        self._output_names = minibatch.get_minibatch_blob_names()
        self._shuffle_roidb_inds()
        self.create_threads()
        
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb. Not thread safe."""
        if True: #cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]

            horz_inds = np.random.permutation(horz_inds)
            vert_inds = np.random.permutation(vert_inds)
            mb = 2 #cfg.TRAIN.IMS_PER_BATCH
            horz_inds = horz_inds[:(len(horz_inds) // mb) * mb]
            vert_inds = vert_inds[:(len(vert_inds) // mb) * mb]
            inds = np.hstack((horz_inds, vert_inds))

            inds = np.reshape(inds, (-1, mb))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._perm = deque(self._perm)
        self._cur = 0
        
    def create_threads(self):
        # Create mini-batch loader threads, each of which builds mini-batches
        # and places them into a queue in CPU memory
        
        self.minibatch_loader_thread()
        
        enqueue_blob_names = self.create_blobs_queues()
        
        self.enqueue_blobs_thread(1, enqueue_blob_names)
    
    def minibatch_loader_thread(self):
        """Load mini-batches and put them onto the mini-batch queue."""
       
        blobs = self.get_next_minibatch()
                
        ordered_blobs = OrderedDict()
        
        for key in self.get_output_names():     
            ordered_blobs[key] = blobs[key]
               
               
               
    def get_output_names(self):
        return self._output_names

    def create_blobs_queues(self):
        return self.create_enqueue_blobs()
   
    def create_enqueue_blobs(self):
        blob_names = self.get_output_names()
        enqueue_blob_names = [
            '{}_enqueue_{}'.format(b, self._loader_id) for b in blob_names
        ]
        return enqueue_blob_names
        
    def enqueue_blobs_thread(self, gpu_id, blob_names):

        return
        
    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch. Thread safe."""
        valid = False
        while not valid:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs, valid = minibatch.get_minibatch(minibatch_db)
        return blobs
        
   
    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch. Thread safe."""
        
        
        # We use a deque and always take the *first* IMS_PER_BATCH items
        # followed by *rotating* the deque so that we see fresh items
        # each time. If the length of _perm is not divisible by
        # IMS_PER_BATCH, then we end up wrapping around the permutation.
        db_inds = [self._perm[i] for i in range(IMS_PER_BATCH)]
        self._perm.rotate(-IMS_PER_BATCH)
        self._cur += IMS_PER_BATCH
        if self._cur >= len(self._perm):
            self._shuffle_roidb_inds()
        return db_inds