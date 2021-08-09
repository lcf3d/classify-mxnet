import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from mxnet import init, nd
from mxnet.gluon import nn
from gluoncv import utils
from gluoncv.model_zoo import get_model


ctx = mx.cpu()

im_fname = 'C:\\Users\\lv\\Desktop\\work\\classify-data\\train\\02\\2100.jpg'

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
#plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
#plt.show()

net = get_model('MobileNet1.0', pretrained=True)
with net.name_scope():
    net.output = nn.Dense(2)
net.output.initialize(init.Xavier(), ctx = ctx)
net.collect_params().reset_ctx(ctx)
net.hybridize()
net.load_parameters('./mobilenet1.0_glass.params')
pred = net(img.expand_dims(axis=0))

net.export('./classify-model', epoch=0)# export model

class_names = ['0', '1']
ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified as [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))