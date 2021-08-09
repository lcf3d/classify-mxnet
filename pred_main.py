import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='classify.')
    parser.add_argument("--image_file", type=str, default='./classify-data\\train\\01\\2.jpg', help='image file')
    parser.add_argument("--model_path", type=str, default='./model', help="model path")
    return parser.parse_args()

#加载模型
def load_model(model_path):
    json_path = model_path + "/" + "classify-model.json"
    params_path =model_path + "/" + "classify-model.params"
    net=gluon.SymbolBlock.imports(json_path, ['data'], params_path, ctx=mx.cpu())
    return net

#图片处理
def image_transform(im_fname):
    img = image.imread(im_fname)
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = transform_fn(img)
    return img
#分类结果
def classify(im_fname, model_path):
    img = image_transform(im_fname)
    net = load_model(model_path)
    pred = net(img.expand_dims(axis=0))
    return pred


if __name__ == "__main__":
    start = time.time() 
    args = parse_args()

    pred = classify(args.image_file, args.model_path)

    class_names = ['0', '1']  #0表玻璃层，1其他层
    ind = nd.argmax(pred, axis=1).astype('int')
    print('分类结果 [%s], 概率 %.3f.'%
            (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar())) 
    
    time_s = "time %.2f sec" % (time.time() - start)
    print(time_s)
    print("classify done!")
