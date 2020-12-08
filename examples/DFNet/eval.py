# coding: utf-8
import numpy as np
from PIL import Image
import scipy.io as sio
import os
import cv2
import time

caffe_root = '../../'
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L

print("import caffe success")


def net_deploy(deploy_prototxt, model):
    from model.dfnet import dfnet

    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])

    dfnet(n, is_train=False)
    n.sigmoid_edge = L.Sigmoid(n.edge_pfuse)
    n.sigmoid_edge1 = L.Sigmoid(n.edge_p1)
    n.sigmoid_edge2 = L.Sigmoid(n.edge_p2)

    with open(deploy_prototxt, 'w') as f:
        f.write(str(n.to_proto()))  ## write network

    net = caffe.Net(deploy_prototxt,
                    model,
                    caffe.TEST)
    return net

## should change the [model path] + [save_path] + [import module]
data_root = '../../data/PIOD/Augmentation/'
save_root = 'Output/dfnet/'
model = 'snapshot/dfnet_iter_30000.caffemodel'

deploy_prototxt = 'dfnet_eval.prototxt'

# load net
caffe.set_mode_gpu()
caffe.set_device(3)
net = net_deploy(deploy_prototxt, model)

save_root = os.path.join(save_root, 'PIOD')
if not os.path.exists(save_root):
    os.mkdir(save_root)


with open(data_root + 'test.lst') as f:
    test_lst = f.readlines()

test_lst = [x.strip() for x in test_lst]

im_lst = []
gt_lst = []

for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    im_lst.append(in_)

start_time = time.time()
for idx in range(0, len(test_lst)):
    print(idx)
    im_ = im_lst[idx]
    im_ = im_.transpose((2, 0, 1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im_.shape)
    net.blobs['data'].data[...] = im_
    # run net and take argmax for prediction
    net.forward()

    edgemap = net.blobs['sigmoid_edge'].data[0][0, :, :]
    orimap = net.blobs['unet1b_ori'].data[0][0, :, :]


    edge_ori = {}
    edge_ori['edge'] = edgemap
    edge_ori['ori'] = orimap
    # plt.imshow(edgemap)
    # plt.show()
    cv2.imwrite(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.png', edgemap * 255)
    sio.savemat(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.mat', {'edge_ori': edge_ori})

diff_time = time.time() - start_time
print('Detection took {:.3f}s per image'.format(diff_time / len(test_lst)))
