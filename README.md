# EVA-8_Phase-1_Assignment-10
This is the assignment of 10th session in Phase-1 of EVA-8 from TSAI
## Introduction

### Objective
This assignment has two part objective:
1. Perform experimentation with Conv mixer model. Test this architecture with our custom writtern architecture, etc.
2. Re-write ViT code such that all the linear layers are replaced by Convolution layers in attention mechanism. 
### Repository structure
Three are three folder containing notebook:
- ConvMixer_experimentation_1: Contains notebook with first experimentation in which model is re-written and custom wrapper is used.
- ConvMixer_expermentation_2: Contains notebbook with second experimentation in which custom wrapper is changed with the training code written in class.
- ViT_with_convolutaion: Contains notebook where ViT model is written in such a way that it does not contain any linear layers.
## Data representation
In this assignment I am using [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) with this dataset I am applying following augmentation on the top:
1. `RandomCrop` - Cropping 32x32 patches from the input image after giving a padding of 4.
1. `HorizontalFlip` - Fliping the image along horizontal axis.
3. `CoarseDropOut` - Overlay a rectangle patch(half the size of original image) on a image randomly. (simulate object hindarence)
6. `Normalize` - Normalize image i.e. zero centring (zero mean) and scaling (one std)

Below is the graph representing the input training dataset after appling all augmentations.
![Alt text](ViT_with_convolution/data_6.png?raw=true "model architecture")
## ConvMixer implementation
## ViT based on convolution
### ViT introduction
## Result & Conclusion
