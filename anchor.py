import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
#from . import mask as maskUtils
import os
from collections import defaultdict
import sys
import scipy.sparse


import Segms
import bbox
from COCO import COCO
from JsonDataset import JsonDataset


dataset_name = 'wheat2019_anchor_test.json'

ds = JsonDataset(dataset_name)
roidb = ds.get_roidb(
    gt=True
)

print(roidb)