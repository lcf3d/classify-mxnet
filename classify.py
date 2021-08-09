import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer learning on MINC-2500 dataset')
    parser.add_argument('--data', type=str, default='./minc-2500-tiny', help='directory for the prepared data folder')
   
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int, help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=0, type=int, help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--epochs', default=40, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float, help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='10,20,30', type=str, help='list of learning rate decay epochs as in str')
    
    return parser.parse_args()

if __name__ == "__main__":

    
    args = parse_args()

