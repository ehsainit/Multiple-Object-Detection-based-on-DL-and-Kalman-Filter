# Real-Time Multiple Objects Tracking DLC-based Framework 

## Introduction

This repository contains code for Realtime Mutiple Object Tracking using [DLC2](https://github.com/AlexEMG/DeepLabCut)  Detectors.
Despite, the recent remarkable progress in object detection deep-learning-based-systems, many of them do not track across time, but rather process each frame individually. Furthermore, the absence of computational efficiency and important latencies inspired taking local motion estimation and thus, achieving higher performance and lower latency. Read the report for more
detailed information.


## Dependencies
The code is compatible with Python 3. The following dependencies are needed to run the framework:
* NumPy
* skimage
* scipy
* OpenCV
* Additionally, feature generation requires TensorFlow (>= 1.1).

## Installation

First, clone the repository:
```
git clone https://github.com/nwojke/deep_sort.git
```
Then, follow the instruction here to install an anconda enviroment:
[here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md).

*NOTE:* for realtime tracking you will need to use the GPU for faster inference.

## Run 
```
python3 main.py [path to config file of the trained dataset] [videos or number of connected device]
```

## Training the model

To train, please follow the instructions on the DLC2 repo:
[DeepLabCut](https://github.com/AlexEMG/DeepLabCut) 

