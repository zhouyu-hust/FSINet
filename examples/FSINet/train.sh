#!/bin/bash
set -x

weights="../../models/resnet/resnet50.caffemodel"
solver="FSINet.prototxt"
../../build/tools/caffe.bin train -solver="$solver" -weights="$weights" -gpu 3

