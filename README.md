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
ConvMixer is a novel image classification architecture that replaces the standard convolutional layers with a combination of depthwise convolutions and channel mixing operations. The architecture consists of multiple stages, where each stage has a set of depthwise convolutions followed by a channel mixing operation. The depthwise convolutions operate independently on each channel of the input feature map, while the channel mixing operation mixes the information across all channels. ConvMixer is computationally efficient since depthwise convolutions are cheaper than standard convolutions, resulting in fewer parameters and faster training.
here is the github link for the original implimentation - [github link](https://github.com/locuslab/convmixer)
![Alt text](ConvMixer_experimentation_1/mixer.png?raw=true "model architecture")

### ConvMixer_experimentation_1
 - In this experimentation model architecture is written in more elaborate manner.
 - Here custom implemented wrapper is used for training. [github link for wrapper](https://github.com/devdastl/eva8_source).
 - Below is the model class
 ```
class residual(nn.Module):
    def __init__(self, res_block):
        super().__init__()
        self.res_block = res_block

    def forward(self, x):
        return self.res_block(x) + x

class MixerModel():
    def __init__(self, dim, depth, kernel_size=5, patch_size=2, n_classes=10):
        self.depth = depth
        self.dim = dim
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.n_classes = n_classes

    def depth_wise(self):
        return nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=self.kernel_size, groups=self.dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )
    def point_wise(self):
        return nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )

    def get_model(self):
        embedding_prep = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )

        depth_wise = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=self.kernel_size, groups=self.dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )   

        point_wise =  nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )

        mixer_block = [nn.Sequential(
            residual(self.depth_wise()),
            self.point_wise()
        ) for i in range(self.depth)]

        model = nn.Sequential(
            embedding_prep,
            *mixer_block,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.dim, self.n_classes)
        )

        return model

 ```
  - Results:
   ```Loss=0.5845258235931396 Batch_id=97 Accuracy=79.38: 100%|██████████| 98/98 [00:59<00:00,  1.65it/s]```
### ConvMixer_experimentation_2
 - In this experimentation, we are using same elaborate model defination.
 - Here instead of custom wrapper, we are using training code discussed in class.
 - Results: ``` ConvMixer: Epoch: 24 | Train Acc: 0.9283, Test Acc: 0.9095, Time: 27.7, lr: 0.000000```

## ViT based on convolution
### ViT introduction

## Result & Conclusion
