import argparse
import glob
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.core.fromnumeric import size
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
import re
import mlflow
logger = logging.getLogger(__name__)


class Train():
    def __init__(self,weights='./weights/yolov5s.pt',cfg='./models/yolov5s.yaml',data='./datasets/obj.yaml',
                 img_size=416,batch_size=10,epochs=10):
        self.weights= weights 
        self.cfg    = cfg 
        self.data   = data 
        self.hyp='data/hyp.scratch.yaml'
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size =img_size 
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        self.resume= False
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        self.bucket=''
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        self.name = "" 
        self.device=''
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        self.local_rank=-1
        self.logdir='runs/'
        self.workers=8
        