#!/bin/bash
set -x

weights="../../models/resnet/resnet50.caffemodel"
solver="solver.prototxt"
../../build/tools/caffe.bin train -solver="$solver" -weights="$weights" -gpu 3

