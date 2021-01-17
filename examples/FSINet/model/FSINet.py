import sys
sys.path.append('python')

caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L, params as P

## False if TRAIN, True if TEST
bn_global_stats = False

def _conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    '''Helper to build a conv -> BN -> relu block.
    '''
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale

def _conv_bn_scale_relu(bottom, nout, bias_term=False, **kwargs):
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu

def _deconv_bn_scale_relu(bottom, nout, kernel_size, stride, pad, bias_term=False):
    ## just a bilinear upsample (lr_mult=0)
    global bn_global_stats

    conv = L.Deconvolution(bottom,
        convolution_param=dict(num_output=nout, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term,
        weight_filler={"type": "bilinear"}), param=[dict(lr_mult=0)])
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu

def _conv_relu(bottom, nout, bias_term=False, **kwargs):
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,**kwargs)
    out_relu = L.ReLU(conv, in_place=True)

    return out_relu, conv

def upsample_layer(bottom, uprate, in_dim):
    up_ = L.Deconvolution(bottom, convolution_param=dict(num_output=in_dim, kernel_size=uprate*2,
                                                                       group=in_dim,
                                                                       stride=uprate, pad=0, bias_term=False,
                                                                       weight_filler={"type": "bilinear"}),
                                                param=[dict(lr_mult=0)])
    return up_



def _aspp_block(n, bottom, dil_rates=[1,6,12,18], out_dim=8):
    n.relu_aspp_1, n.aspp1 = _conv_relu(bottom, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})
    n.relu_6_1, n.conv6_1 = _conv_relu(n.relu_aspp_1, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_2, n.aspp_2 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[0], pad=dil_rates[0],
                                         weight_filler={"type": "msra"})
    n.relu_6_2, n.conv6_2 = _conv_relu(n.relu_aspp_2, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_3, n.aspp_3 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[1], pad=dil_rates[1],
                                         weight_filler={"type": "msra"})
    n.relu_6_3, n.conv6_3 = _conv_relu(n.relu_aspp_3, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_4, n.aspp_4 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[2], pad=dil_rates[2],
                                         weight_filler={"type": "msra"})
    n.relu_6_4, n.conv6_4 = _conv_relu(n.relu_aspp_4, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    concat_layers = [n.relu_6_1, n.relu_6_2, n.relu_6_3, n.relu_6_4]
    n.aspp_concat = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))


def _res_refinement(name, n, bottom, in_dim):
    '''Basic ResNet block.
    '''
    n['res{}_bottom'.format(name)], _, _, n['res{}_bottom_relu'.format(name)] = \
        _conv_bn_scale_relu(bottom, nout=in_dim, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n['res{}_bottom_relu'.format(name)], in_dim, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], in_dim, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)


    n[res] = L.Eltwise(n['res{}_bottom_relu'.format(name)], n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

    return n[relu]

def side_convert(n):
    inplanes = 8

    n['res2c_1'], n['res2c_1_bn'], n['res2c_1_scale'], n['res2c_1_relu'] = \
        _conv_bn_scale_relu(n.res2c_relu, nout=inplanes, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    n['res3d_1'], n['res3d_1_bn'], n['res3d_1_scale'], n['res3d_1_relu'] = \
        _conv_bn_scale_relu(n.res3d_relu,nout=inplanes, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    n['res4f_1'], n['res4f_1_bn'], n['res4f_1_scale'], n['res4f_1_relu'] = \
        _conv_bn_scale_relu(n.res4f_relu, nout=inplanes, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    n['res5c_1'], n['res5c_1_bn'], n['res5c_1_scale'], n['res5c_1_relu'] = \
        _conv_bn_scale_relu(n.res5c_relu, nout=inplanes, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    n.res3d_1_up = upsample_layer(n['res3d_1_relu'], uprate=2, in_dim=inplanes)
    n.res4f_1_up = upsample_layer(n['res4f_1_relu'], uprate=4, in_dim=inplanes)
    n.res5c_1_up = upsample_layer(n['res5c_1_relu'], uprate=4, in_dim=inplanes)

    n.res3d_1_crop = L.Crop(n.res3d_1_up, n.res2c_1_relu, crop_param={'axis': 2})
    n.res4f_1_crop = L.Crop(n.res4f_1_up, n.res2c_1_relu, crop_param={'axis': 2})
    n.res5c_1_crop = L.Crop(n.res5c_1_up, n.res2c_1_relu, crop_param={'axis': 2})


    ## start fusion
    n.fuse1 = L.Concat(n.res2c_1_relu, n.res3d_1_crop, concat_param=dict(concat_dim=1))
    n.fuse1_refine, _, _, n.fuse1_refine_relu = \
        _conv_bn_scale_relu(n.fuse1, nout=inplanes, bias_term=True, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.fuse2 = L.Concat(n.fuse1_refine_relu, n.res4f_1_crop, concat_param=dict(concat_dim=1))
    n.fuse2_refine, _, _, n.fuse2_refine_relu = \
        _conv_bn_scale_relu(n.fuse2, nout=inplanes, bias_term=True, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.fuse3 = L.Concat(n.fuse2_refine_relu, n.res5c_1_crop, concat_param=dict(concat_dim=1))
    n.fuse3_refine0, _, _, n.fuse3_refine0_relu = \
        _conv_bn_scale_relu(n.fuse3, nout=inplanes, bias_term=True, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.fuse3_res_up = upsample_layer(n.fuse3_refine0_relu, uprate=4, in_dim=inplanes)
    n.fuse3_1_crop = L.Crop(n.fuse3_res_up, n.data, crop_param={'axis': 2})

    n.fuse3_refine, _, _, n.fuse3_refine_relu = \
        _conv_bn_scale_relu(n.fuse3_1_crop, nout=16, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})


def cascade_branch(n, bottom, branch):
    """ For edge and ori output (task: edge or ori)
        edge and ori has same branch [8,8,8,8,4,1]
    """
    conv = 'unet1a_plain1a_conv_{}'.format(branch)
    bn = 'unet1a_plain1a_bn_{}'.format(branch)
    scale = 'unet1a_plain1a_scale_{}'.format(branch)
    relu1 = 'unet1a_plain1a_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu1] = _conv_bn_scale_relu(bottom, nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1a_plain1b_conv_{}'.format(branch)
    bn = 'unet1a_plain1b_bn_{}'.format(branch)
    scale = 'unet1a_plain1b_scale_{}'.format(branch)
    relu2 = 'unet1a_plain1b_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu2] = _conv_bn_scale_relu(n[relu1], nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})
    conv = 'unet1b_plain1a_conv_{}'.format(branch)
    bn = 'unet1b_plain1a_bn_{}'.format(branch)
    scale = 'unet1b_plain1a_scale_{}'.format(branch)
    relu4 = 'unet1b_plain1a_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu4] = _conv_bn_scale_relu(n[relu2], nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_plain1b_conv_{}'.format(branch)
    bn = 'unet1b_plain1b_bn_{}'.format(branch)
    scale = 'unet1b_plain1b_scale_{}'.format(branch)
    relu5 = 'unet1b_plain1b_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu5] = _conv_bn_scale_relu(n[relu4], nout=4, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})

def cascade_output(n, bottom, task):
    ## pre-process the input feature, relu1 is origin edge feature, and relu2~4 is residual
    cascade_branch(n, bottom, branch='edge_branch1')

    ## init thick edge prediction and residual
    n.edge_p1 = L.Convolution(n.unet1b_plain1b_relu_edge_branch1, num_output=1, bias_term=True, kernel_size=1,
                              stride=1, pad=0, weight_filler={"type": "msra"},
                              bias_filler={"type": "constant", "value": 0.0})

    ## --------------
    branch = 'edge_branch2'

    conv = 'unet1a_plain1a_conv_{}'.format(branch)
    bn = 'unet1a_plain1a_bn_{}'.format(branch)
    scale = 'unet1a_plain1a_scale_{}'.format(branch)
    relu1 = 'unet1a_plain1a_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu1] = _conv_bn_scale_relu(bottom, nout=8, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})
    n.cas_1 = L.Eltwise(n[relu1], n.unet1a_plain1a_relu_edge_branch1)

    conv = 'unet1a_plain1b_conv_{}'.format(branch)
    bn = 'unet1a_plain1b_bn_{}'.format(branch)
    scale = 'unet1a_plain1b_scale_{}'.format(branch)
    relu2 = 'unet1a_plain1b_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu2] = _conv_bn_scale_relu(n.cas_1, nout=8, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})
    n.cas_1d5 = L.Eltwise(n[relu2], n.unet1a_plain1b_relu_edge_branch1)

    conv = 'unet1b_plain1a_conv_{}'.format(branch)
    bn = 'unet1b_plain1a_bn_{}'.format(branch)
    scale = 'unet1b_plain1a_scale_{}'.format(branch)
    relu4 = 'unet1b_plain1a_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu4] = _conv_bn_scale_relu(n.cas_1d5, nout=8, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})
    n.cas_2 = L.Eltwise(n[relu4], n.unet1b_plain1a_relu_edge_branch1)

    conv = 'unet1b_plain1b_conv_{}'.format(branch)
    bn = 'unet1b_plain1b_bn_{}'.format(branch)
    scale = 'unet1b_plain1b_scale_{}'.format(branch)
    relu5 = 'unet1b_plain1b_relu_{}'.format(branch)
    n[conv], n[bn], n[scale], n[relu5] = _conv_bn_scale_relu(n.cas_2, nout=4, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})
    n.cas_3 = L.Eltwise(n[relu5], n.unet1b_plain1b_relu_edge_branch1)

    n.edge_p2 = L.Convolution(n.cas_3, num_output=1, bias_term=True, kernel_size=1,
                              stride=1, pad=0, weight_filler={"type": "msra"},
                              bias_filler={"type": "constant", "value": 0.0},
                              )



    n.edge_feat_concat = L.Concat(n.unet1b_plain1b_relu_edge_branch1,
                                  n.unet1b_plain1b_relu_edge_branch2,
                                  concat_param=dict(concat_dim=1))

    conv = '{}_pfuse'.format(task)
    n[conv] = L.Convolution(n.edge_feat_concat, num_output=1, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"},
                            bias_filler={"type": "constant", "value": 0.0},
                            )


def output_strip_branch(n, bottom, task):
    conv = 'unet1b_plain1a_conv_{}'.format(task)
    bn = 'unet1b_plain1a_bn_{}'.format(task)
    scale = 'unet1b_plain1a_scale_{}'.format(task)
    relu4 = 'unet1b_plain1a_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu4] = _conv_bn_scale_relu(bottom, nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_plain1b_conv_{}'.format(task)
    bn = 'unet1b_plain1b_bn_{}'.format(task)
    scale = 'unet1b_plain1b_scale_{}'.format(task)
    relu5 = 'unet1b_plain1b_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu5] = _conv_bn_scale_relu(n[relu4], nout=4, bias_term=True,
                                                             kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_{}'.format(task)
    n[conv] = L.Convolution(n[relu5], num_output=1, bias_term=True, kernel_size=1,
                                                             stride=1, pad=0, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})


def _resnet_block(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    if branch1:  
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

def _resnet_block2(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    mul = 2

    if branch1:  
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, mul*nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], mul*nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

def _resnet_dilation_block(name, n, bottom, nout, branch1=False, initial_stride=2, dil_rate=2):
    '''Basic ResNet block.
    '''
    if branch1:
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride,
            pad=0, weight_filler={"type": "msra"})
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    # dilation
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=dil_rate, weight_filler={"type": "msra"}, dilation=dil_rate)
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)


def resnet50_branch(n, bottom):
    '''ResNet 50 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=True, kernel_size=7, pad=3, stride=2, weight_filler={"type": "msra"})
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # stage 2
    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64) # res2c_relu

    # stage 3
    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128) # res3d_relu

    # stage 4
    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256) # res4f_relu

    # stage 5
    _resnet_dilation_block('5a', n, n.res4f_relu, 512, branch1=True, initial_stride=1, dil_rate=2)
    _resnet_dilation_block('5b', n, n.res5a_relu, 512, dil_rate=4)
    _resnet_dilation_block('5c', n, n.res5b_relu, 512, dil_rate=8) # res5c_relu


def decoder_path(n, bottom):
    n.unet3a_deconv_relu= upsample_layer(bottom, uprate=4, in_dim=256)

    crop_bottoms = [n.unet3a_deconv_relu, n.res2c_relu]
    n.unet3a_crop = L.Crop(*crop_bottoms, crop_param={'axis': 2, 'offset': 1})  ## crop 1/4

    concat_layers = [n.unet3a_crop, n.res2c_relu]
    n.unet3a_concat = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))  ## concat with res2c

    _resnet_block('6a', n, n.unet3a_concat, 128)
    _resnet_block2('6b', n, n.res6a_relu, 8, branch1=True, initial_stride=1)  ## res 6

    n.unet1a_deconv_relu = upsample_layer(n.res6b_relu, uprate=4, in_dim=16)

    crop_bottoms = [n.unet1a_deconv_relu, n.data]
    n.unet1a_crop = L.Crop(*crop_bottoms, crop_param={'axis': 2, 'offset': 1})  ## crop 1 -> occlusion cue


def filter_block(n, bottom, dil_rate):
    n.res5c_dil, n.res5c_dil_bn, n.res5c_dil_scale, n.res5c_dil_relu = \
        _conv_bn_scale_relu(bottom, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=dil_rate, weight_filler={"type": "msra"},dilation=dil_rate) # dilation conv

    n.res5c_a1, n.res5c_a1_bn, n.res5c_a1_scale, n.res5c_a1_relu = \
        _conv_bn_scale_relu(n.res5c_dil_relu, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.res5c_a2, n.res5c_a2_bn, n.res5c_a2_scale, n.res5c_a2_relu = \
        _conv_bn_scale_relu(n.res5c_a1_relu, nout=16, bias_term=False, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})


def edge_path(n, bottom):
    filter_block(n, bottom, dil_rate=2)

    n.res5c_up2_relu = upsample_layer(n.res5c_a2_relu, uprate=16, in_dim=16)

    crop_bottoms = [n.res5c_up2_relu, n.data]
    n.res5c_crop2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1


    ## -> from data directly
    n.unet1a_conv1_edge, n.unet1a_bn_conv1_edge, n.unet1a_scale_conv1_edge, n.unet1a_conv1_relu_edge = \
        _conv_bn_scale_relu(n.data, nout=8, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet1a_conv2_edge, n.unet1a_bn_conv2_edge, n.unet1a_scale_conv2_edge, n.unet1a_conv2_relu_edge = \
        _conv_bn_scale_relu(n.unet1a_conv1_relu_edge, nout=4, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet1a_conv3_edge, n.unet1a_bn_conv3_edge, n.unet1a_scale_conv3_edge, n.unet1a_conv3_relu_edge = \
        _conv_bn_scale_relu(n.unet1a_conv2_relu_edge, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet2a_conv1_edge, n.unet2a_bn_conv1_edge, n.unet2a_scale_conv1_edge, n.unet2a_conv1_relu_edge = \
        _conv_bn_scale_relu(n.data, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet1a = L.Eltwise(n.unet1a_conv3_relu_edge, n.unet2a_conv1_relu_edge)

    ## side out
    side_convert(n)


def _mcl(n, bottom, dil_rates=[2, 3, 6], out_dim=256):
    n.aspp_1, _, _, n.relu_aspp_1 = _conv_bn_scale_relu(bottom, nout=out_dim, bias_term=False,
                                                        kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    n.aspp_2, _, _, n.relu_aspp_2, = _conv_bn_scale_relu(bottom, nout=out_dim, kernel_size=3,
                                                         dilation=dil_rates[0],
                                                         pad=dil_rates[0], stride=1, weight_filler={"type": "msra"})
    n.aspp_3, _, _, n.relu_aspp_3, = _conv_bn_scale_relu(bottom, nout=out_dim, kernel_size=3,
                                                         dilation=dil_rates[1],
                                                         pad=dil_rates[1], stride=1, weight_filler={"type": "msra"})
    n.aspp_4, _, _, n.relu_aspp_4, = _conv_bn_scale_relu(bottom, nout=out_dim, kernel_size=3,
                                                         dilation=dil_rates[2],
                                                         pad=dil_rates[2], stride=1, weight_filler={"type": "msra"})

    concat_layers = [n.relu_aspp_1, n.relu_aspp_2, n.relu_aspp_3, n.relu_aspp_4]
    n.aspp_concat = L.Concat(*concat_layers, concat_param=dict(concat_dim=1))

    n.aspp_refine, _, _, n.aspp_refine_relu = _conv_bn_scale_relu(n.aspp_concat, nout=out_dim, kernel_size=1,
                                                                  stride=1, weight_filler={"type": "msra"})

def ori_path(n, bottom):
    ## aspp module
    _mcl(n, bottom)

    n.aspp_reduce, _, _, n.aspp_reduce_relu = \
        _conv_bn_scale_relu(n.aspp_refine_relu, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.aspp_up2_relu = upsample_layer(n.aspp_reduce_relu, uprate=4, in_dim=16)

    crop_bottoms = [n.aspp_up2_relu, n.res6b_relu]
    n.aspp_crop2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1

    n.ori_concat1 = L.Concat(n.aspp_crop2, n.res6b_relu)

    task = 'ori'
    n['{}_v1'.format(task)], _, _, n['{}_v1_relu'.format(task)] = _conv_bn_scale_relu(n.ori_concat1, nout=16,
                                                                                      bias_term=False,
                                                                                      kernel_size=[3, 7], stride=1,
                                                                                      pad=[1, 3],
                                                                                      weight_filler={"type": "msra"})
    n['{}_h1'.format(task)], _, _, n['{}_h1_relu'.format(task)] = _conv_bn_scale_relu(n.ori_concat1, nout=16,
                                                                                      bias_term=False,
                                                                                      kernel_size=[7, 3], stride=1,
                                                                                      pad=[3, 1],
                                                                                      weight_filler={"type": "msra"})

    n['{}_strip_concat'.format(task)] = L.Eltwise(n['{}_h1_relu'.format(task)], n['{}_v1_relu'.format(task)])

    conv = 'unet1a_plain1a_conv_{}'.format(task)
    bn = 'unet1a_plain1a_bn_{}'.format(task)
    scale = 'unet1a_plain1a_scale_{}'.format(task)
    relu1 = 'unet1a_plain1a_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu1] = _conv_bn_scale_relu(n['{}_strip_concat'.format(task)],
                                                             nout=16, bias_term=True,
                                                             kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})

    n.aspp_up3_relu = upsample_layer(n.unet1a_plain1a_relu_ori, uprate=4, in_dim=16)

    crop_bottoms = [n.aspp_up3_relu, n.data]
    n.aspp_crop3 = L.Crop(*crop_bottoms, crop_param={'axis': 2})



def FSINet(n,is_train=True):
    global bn_global_stats
    bn_global_stats = False if is_train else True

    resnet50_branch(n, n.data) ## resnet50 backbone

    ## plain 6
    n.plain6a_conv, n.plain6a_bn, n.plain6a_scale, n.plain6a_relu = \
            _conv_bn_scale_relu(n.res5c_relu, nout=256,bias_term=False, kernel_size=3,
            stride=1, pad=1, weight_filler={"type": "msra"})

    n.plain6b_conv, n.plain6b_bn, n.plain6b_scale, n.plain6b_relu = \
        _conv_bn_scale_relu(n.plain6a_relu, nout=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.plain7a_conv, n.plain7a_bn, n.plain7a_scale, n.plain7a_relu = \
        _conv_bn_scale_relu(n.res5c_relu, nout=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    ## plain6b_relu is the origin of three path -----

    ## decoder path
    decoder_path(n, n.plain6b_relu)

    ## edge path
    edge_path(n, n.plain6b_relu)
    ## ori path
    ori_path(n, n.plain7a_relu)



    ## edge output ----------------------
    concat_layers = [n.unet1a_crop, n.unet1a, n.res5c_crop2, n.fuse3_refine_relu]
    n.unet1a_concat_edge = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))

    n.edge_reduce, _, _, n.edge_reduce_relu = \
        _conv_bn_scale_relu(n.unet1a_concat_edge, nout=16, bias_term=False, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})
    cascade_output(n, n.edge_reduce_relu, task='edge')

    ## ori output -----------------------
    output_strip_branch(n, n.aspp_crop3, task='ori')


if __name__ == '__main__':
    n = caffe.NetSpec()
    # n.data = L.DummyData(shape=[dict(dim=[1, 3, 224, 224])])
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])


    ofnet(n, is_train=True)
    with open('ofnet_example.prototxt', 'w') as f:
        f.write(str(n.to_proto()))










