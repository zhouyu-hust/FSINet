import sys
sys.path.append('python')

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
print('import caffe success')

import caffe
from caffe import layers as L, params as P

from model.FSINet import FSINet

def write_network(data_path="../../data/PIOD/Augmentation/train_pair_320x320.lst", batch_size=5):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageLabelmapData(include={'phase': 0},  ## 0-TRAIN 1-TEST
                                          image_data_param={
                                              'source': data_path,
                                              'batch_size': batch_size,
                                              'shuffle': True,
                                              'new_height': 0,
                                              'new_width': 0,
                                              'root_folder': "",
                                              'data_type': "h5"},
                                          transform_param={
                                              'mirror': False,
                                              'crop_size': 320,
                                              'mean_value': [104.006988525, 116.668769836, 122.678916931]
                                          },
                                          ntop=2)

    n.label_edge, n.label_ori = L.Slice(n.label, slice_param={'slice_point': 1}, ntop=2)

    FSINet(n, is_train=True)

    loss_bottoms = [n.edge_p1, n.label_edge]
    n.edge_loss1 = L.ClassBalancedSigmoidCrossEntropyAttentionLoss(*loss_bottoms,
                                                                  loss_weight=1.0,
                                                                  attention_loss_param={'beta': 4.0,
                                                                                        'gamma': 0.5})

    loss_bottoms = [n.edge_p2, n.label_edge]
    n.edge_loss2 = L.SigmoidCrossEntropyLoss(*loss_bottoms,
                                             loss_weight=0.3)


    loss_bottoms = [n.edge_pfuse, n.label_edge]
    n.edge_loss = L.ClassBalancedSigmoidCrossEntropyAttentionLoss(*loss_bottoms,
                                                                  loss_weight=2.1,
                                                                  attention_loss_param={'beta': 4.0,
                                                                                        'gamma': 0.5})

    loss_bottoms = [n.unet1b_ori, n.label_ori, n.label_edge]
    n.ori_loss = L.OrientationSmoothL1Loss(*loss_bottoms, loss_weight=1.0, smooth_l1_loss_param={'sigma': 3.0})

    with open('FSINet.prototxt', 'w') as f:
        f.write(str(n.to_proto()))  ## write network



def train(initmodel, gpu):
    write_network()
    print('write network in prototxt')


if __name__ == '__main__':
    train(initmodel='ResNet-50-model.caffemodel', gpu=0)
