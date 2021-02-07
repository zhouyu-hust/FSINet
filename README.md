# Occlusion Relationship Reasoning with A Feature Separation and Interaction Network

Code for the paper Occlusion Relationship Reasoning with A Feature Separation and Interaction Network. Submit to TPAMI.

Authors: Yu Zhou, Rui Lu, Feng Xue,Yuzhe Gao, and Xiaojie Guo

### Introduction

In this work, we propose the Feature Separation and Interaction Network (FSINet) to present the speciality and complementary between the occlusion boundary detection and the occlusion orientation estimation. 

The occlusion boundary path contains an Image-level Cue Extractor (ICE) to capture rich location information of the boundary, a Detail-perceived Semantic Feature Extractor (DSFE), and a Contextual Correlation Extractor (CCE) to acquire refined semantic messages of objects. A Dual-flow Cross Detector (DCD) is customized to eliminate false-positive boundaries. 

The occlusion orientation estimation path contains a Scene Context Learner (SCL) is designed to capture the depth order cue around the boundary, and two strip convolutions are built to judge the depth order between objects. 

The shared decoder supplies the feature interaction, which plays a key role in exploiting the complementary of the two paths. 

![FSINet](images/FSINet_arch.png)

<div align=center id="model">Network Architecture</div>

## Data Preparation

The Data Preparation and Evaluation are following Guoxia Wang with his [DOOBNet](https://github.com/GuoxiaWang/DOOBNet). Thanks for his valuable work.

#### PASCAL Instance Occlusion Dataset (PIOD)

You may download the dataset original images from [PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) and annotations from [here](https://drive.google.com/file/d/0B7DaWBKShuMBSkZ6Mm5RVmg5ck0/view?usp=sharing). Then you should copy or move `JPEGImages` folder in PASCAL VOC 2010 and `Data` folder and val\_doc_2010.txt in PIOD to `data/PIOD/`. You will have the following directory structure:
```
PIOD
|_ Data
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ JPEGImages 
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ val_doc_2010.txt
```

Now, you can use data convert tool to augment and generate HDF5 format data for DFNet. 
```
mkdir data/PIOD/Augmentation

python doobscripts/doobnet_mat2hdf5_edge_ori.py \
--dataset PIOD \
--label-dir data/PIOD/Data \
--img-dir data/PIOD/JPEGImages \
--piod-val-list-file data/PIOD/val_doc_2010.txt \
--output-dir data/PIOD/Augmentation
```

#### BSDS ownership

For BSDS ownership dataset, you may download the dataset original images from [BSDS300](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz) and annotations from [here](https://drive.google.com/open?id=0B7DaWBKShuMBd3Z0Vmk3UkZxcUU). Then you should copy or move `BSDS300` folder in BSDS300-images and `trainfg` and `testfg` folder in BSDS\_theta to `data/BSDSownership/`. And you will have the following directory structure:
```
BSDSownership
|_ trainfg
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ testfg
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ BSDS300
|  |_ images
|     |_ train
|        |_ <id-1>.jpg
|        |_ ...
|        |_ <id-n>.jpg
|     |_ ...
|  |_ ...
```
Note that BSDS ownership's test set are split from 200 train images (100 for train, 100 for test). More information you can check ids in `trainfg` and `testfg` folder and ids in `BSDS300/images/train` folder, or refer to [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/fg/fgdata.tar.gz)

Run the following code for BSDS ownership dataset. 
```
mkdir data/BSDSownership/Augmentation

python doobscripts/doobnet_mat2hdf5_edge_ori.py \
--dataset BSDSownership \
--label-dir data/BSDSownership/trainfg \
--img-dir data/BSDSownership/BSDS300/images/train \
--bsdsownership-testfg data/BSDSownership/testfg \
--output-dir data/BSDSownership/Augmentation 
```

## Training

Firstly, you need to download the Res50 weight file from [Res50](https://drive.google.com/open?id=1nyGjqSj0LGVsY9iBhsEdo-TXSyROGTgZ) and save `resnet50.caffemodel` to the folder `$DFNET_ROOT/models/resnet/`.

#### PASCAL Instance Occlusion Dataset (PIOD)

For training FSINet on PIOD training dataset, you can run:

```
cd $DFNET_ROOT/examples/DFNet
python write_prototxt.py
bash train.sh
```
When training completed, you need to modify the save model path `model = 'snapshot/dfnet_iter_30000.caffemodel'` in `eval.py` and then run `python eval.py` to get the results on PIOD testing dataset. For comparation, you can also download our trained model from [here](https://pan.baidu.com/s/1RUVQZCCbA5kQJWbaSIxp-g ). (code: jjnt). The testing results are available at [here](https://pan.baidu.com/s/1VV0kwDsfITPey5yCJjLMHg  ). (code: ynqs).


#### BSDS ownership
For training DFNet on BSDS ownership, you can refer the manner as same as PIOD dataset above. The training model is available at [here](https://pan.baidu.com/s/10dIpfIticC1sQUf1qXxjdA ). (code: 5bmf). The testing results are available at [here](https://pan.baidu.com/s/16Sm2VrXBRsR5hIwVkCwU4Q ). (code: 2uni).


## Evaluation

Here we provide the PIOD and the BSDS ownership dataset's evaluation and visualization code in `doobscripts` folder.

**Note that** you need to config the necessary paths or variables. More information please refers to `doobscripts/README.md`.

To run the evaluation:
```
run doobscripts/evaluation/EvaluateOcc.m
```

#### Option
For visualization, to run the script:
```
run doobscripts/visulation/PlotAll.m
```

#### Results

Tab.1 Comparisons on the PIOD dataset with the state-of-the-art methods.

|  Method   |   ODS-E   |   OIS-E   |   AP-E   |   ODS-O   |   OIS-O   |   AP-O   |
| ---- | --- | --- | --- | --- | --- | --- |
| SRF-OCC | .345 | .369 | .207 | .268 | .286 | .152 |
| DOC-HED  | .509 | .532| .468 | .460 | .479 | .405 |
| DOC-DMLFOV | .669 | .684 | .677 | .601 | .611 | .585 |
| DOOBNet | .736 | .746 | .723 | .702 | .712 | .683 |
| OFNet | .751 | .762 | .773 | .718 | .728 | .729 |
| FSINet | .762 | .774 | .779 | .733 | .743 | .738 |

Tab.2 Comparisons on the BSDS ownership dataset with the state-of-the-art methods.

|  Method   |   ODS-E   |   OIS-E   |   AP-E   |   ODS-O   |   OIS-O   |   AP-O   |
| ---- | --- | --- | --- | --- | --- | --- |
| SRF-OCC | .511 | .544 | .442 | .419 | .448 | .337 |
| DOC-HED  | .658 | .685 | .602 | .522 | .545 | .428 |
| DOC-DMLFOV | .579 | .609 | .519 | .463 | .491 | .369 |
| DOOBNet | .647 | .668 | .539 | .555 | .570 | .440 |
| OFNet | .662 | .689 | .585 | .583 | .607 | .501 |
| FSINet | .657 | .692 | .598 | .591 | .620 | .515 |

    
