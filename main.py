import time
import numpy as np
from mxnet import autograd, gluon
import gluoncv as gcv
import cv2
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from mxnet import init, nd
from mxnet.gluon import nn
from gluoncv import utils
from gluoncv.model_zoo import get_model

start = time.time() 


#load model
net=gluon.SymbolBlock.imports('./classify-model.json', ['data'], './classify-model.params', ctx=mx.cpu())

#load image
im_fname = 'C:\\Users\\lv\\Desktop\\work\\classify-data\\train\\01\\3.jpg'

img = image.imread(im_fname)

#plt.imshow(img.asnumpy())
#plt.show()

transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform_fn(img)

pred = net(img.expand_dims(axis=0))


class_names = ['0', '1']
ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified as [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))    
