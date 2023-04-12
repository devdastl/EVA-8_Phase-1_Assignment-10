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
ViT (Vision Transformer) is a state-of-the-art deep learning model for image recognition tasks. It is based on the transformer architecture and is capable of processing images as sequences of tokens. ViT is trained on a large amount of labeled image data and learns to extract features from the image by dividing it into a set of fixed-size patches. These patches are then fed into a series of transformer blocks, which enable the model to learn complex relationships between the image features. ViT has achieved impressive results on a variety of image recognition benchmarks and has shown to be highly effective in transfer learning scenarios. Its success has paved the way for the development of other transformer-based models for computer vision tasks.

### ViT with convolution
- here the code is re-written in such a way that there is no linear layer for attention mechanism.
- Below is the model architecture summary where there is no linear layer

```
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Rearrange-1               [-1, 64, 48]               0
            Conv2d-2           [-1, 512, 64, 1]          24,576
      ModifyConv2d-3              [-1, 64, 512]               0
           Dropout-4              [-1, 65, 512]               0
         LayerNorm-5              [-1, 65, 512]           1,024
            Conv2d-6          [-1, 1536, 65, 1]         786,432
      ModifyConv2d-7             [-1, 65, 1536]               0
           Softmax-8            [-1, 8, 65, 65]               0
            Conv2d-9           [-1, 512, 65, 1]         262,144
     ModifyConv2d-10              [-1, 65, 512]               0
          Dropout-11              [-1, 65, 512]               0
        Attention-12              [-1, 65, 512]               0
          PreNorm-13              [-1, 65, 512]               0
        LayerNorm-14              [-1, 65, 512]           1,024
           Conv2d-15           [-1, 512, 65, 1]         262,144
     ModifyConv2d-16              [-1, 65, 512]               0
             GELU-17              [-1, 65, 512]               0
          Dropout-18              [-1, 65, 512]               0
           Conv2d-19           [-1, 512, 65, 1]         262,144
     ModifyConv2d-20              [-1, 65, 512]               0
          Dropout-21              [-1, 65, 512]               0
      FeedForward-22              [-1, 65, 512]               0
          PreNorm-23              [-1, 65, 512]               0
        LayerNorm-24              [-1, 65, 512]           1,024
           Conv2d-25          [-1, 1536, 65, 1]         786,432
     ModifyConv2d-26             [-1, 65, 1536]               0
          Softmax-27            [-1, 8, 65, 65]               0
           Conv2d-28           [-1, 512, 65, 1]         262,144
     ModifyConv2d-29              [-1, 65, 512]               0
          Dropout-30              [-1, 65, 512]               0
        Attention-31              [-1, 65, 512]               0
          PreNorm-32              [-1, 65, 512]               0
        LayerNorm-33              [-1, 65, 512]           1,024
           Conv2d-34           [-1, 512, 65, 1]         262,144
     ModifyConv2d-35              [-1, 65, 512]               0
             GELU-36              [-1, 65, 512]               0
          Dropout-37              [-1, 65, 512]               0
           Conv2d-38           [-1, 512, 65, 1]         262,144
     ModifyConv2d-39              [-1, 65, 512]               0
          Dropout-40              [-1, 65, 512]               0
      FeedForward-41              [-1, 65, 512]               0
          PreNorm-42              [-1, 65, 512]               0
        LayerNorm-43              [-1, 65, 512]           1,024
           Conv2d-44          [-1, 1536, 65, 1]         786,432
     ModifyConv2d-45             [-1, 65, 1536]               0
          Softmax-46            [-1, 8, 65, 65]               0
           Conv2d-47           [-1, 512, 65, 1]         262,144
     ModifyConv2d-48              [-1, 65, 512]               0
          Dropout-49              [-1, 65, 512]               0
        Attention-50              [-1, 65, 512]               0
          PreNorm-51              [-1, 65, 512]               0
        LayerNorm-52              [-1, 65, 512]           1,024
           Conv2d-53           [-1, 512, 65, 1]         262,144
     ModifyConv2d-54              [-1, 65, 512]               0
             GELU-55              [-1, 65, 512]               0
          Dropout-56              [-1, 65, 512]               0
           Conv2d-57           [-1, 512, 65, 1]         262,144
     ModifyConv2d-58              [-1, 65, 512]               0
          Dropout-59              [-1, 65, 512]               0
      FeedForward-60              [-1, 65, 512]               0
          PreNorm-61              [-1, 65, 512]               0
        LayerNorm-62              [-1, 65, 512]           1,024
           Conv2d-63          [-1, 1536, 65, 1]         786,432
     ModifyConv2d-64             [-1, 65, 1536]               0
          Softmax-65            [-1, 8, 65, 65]               0
           Conv2d-66           [-1, 512, 65, 1]         262,144
     ModifyConv2d-67              [-1, 65, 512]               0
          Dropout-68              [-1, 65, 512]               0
        Attention-69              [-1, 65, 512]               0
          PreNorm-70              [-1, 65, 512]               0
        LayerNorm-71              [-1, 65, 512]           1,024
           Conv2d-72           [-1, 512, 65, 1]         262,144
     ModifyConv2d-73              [-1, 65, 512]               0
             GELU-74              [-1, 65, 512]               0
          Dropout-75              [-1, 65, 512]               0
           Conv2d-76           [-1, 512, 65, 1]         262,144
     ModifyConv2d-77              [-1, 65, 512]               0
          Dropout-78              [-1, 65, 512]               0
      FeedForward-79              [-1, 65, 512]               0
          PreNorm-80              [-1, 65, 512]               0
        LayerNorm-81              [-1, 65, 512]           1,024
           Conv2d-82          [-1, 1536, 65, 1]         786,432
     ModifyConv2d-83             [-1, 65, 1536]               0
          Softmax-84            [-1, 8, 65, 65]               0
           Conv2d-85           [-1, 512, 65, 1]         262,144
     ModifyConv2d-86              [-1, 65, 512]               0
          Dropout-87              [-1, 65, 512]               0
        Attention-88              [-1, 65, 512]               0
          PreNorm-89              [-1, 65, 512]               0
        LayerNorm-90              [-1, 65, 512]           1,024
           Conv2d-91           [-1, 512, 65, 1]         262,144
     ModifyConv2d-92              [-1, 65, 512]               0
             GELU-93              [-1, 65, 512]               0
          Dropout-94              [-1, 65, 512]               0
           Conv2d-95           [-1, 512, 65, 1]         262,144
     ModifyConv2d-96              [-1, 65, 512]               0
          Dropout-97              [-1, 65, 512]               0
      FeedForward-98              [-1, 65, 512]               0
          PreNorm-99              [-1, 65, 512]               0
       LayerNorm-100              [-1, 65, 512]           1,024
          Conv2d-101          [-1, 1536, 65, 1]         786,432
    ModifyConv2d-102             [-1, 65, 1536]               0
         Softmax-103            [-1, 8, 65, 65]               0
          Conv2d-104           [-1, 512, 65, 1]         262,144
    ModifyConv2d-105              [-1, 65, 512]               0
         Dropout-106              [-1, 65, 512]               0
       Attention-107              [-1, 65, 512]               0
         PreNorm-108              [-1, 65, 512]               0
       LayerNorm-109              [-1, 65, 512]           1,024
          Conv2d-110           [-1, 512, 65, 1]         262,144
    ModifyConv2d-111              [-1, 65, 512]               0
            GELU-112              [-1, 65, 512]               0
         Dropout-113              [-1, 65, 512]               0
          Conv2d-114           [-1, 512, 65, 1]         262,144
    ModifyConv2d-115              [-1, 65, 512]               0
         Dropout-116              [-1, 65, 512]               0
     FeedForward-117              [-1, 65, 512]               0
         PreNorm-118              [-1, 65, 512]               0
     Transformer-119              [-1, 65, 512]               0
        Identity-120                  [-1, 512]               0
       LayerNorm-121                  [-1, 512]           1,024
          Linear-122                   [-1, 10]           5,130
================================================================
Total params: 9,480,202
Trainable params: 9,480,202
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 36.10
Params size (MB): 36.16
Estimated Total Size (MB): 72.28
----------------------------------------------------------------
```
 - Results: ``` ViT: Epoch: 24 | Train Acc: 0.6438, Test Acc: 0.6850, Time: 45.7```

