# ADFNet
**Adaptive Dynamic Filtering Network for Image Denoising** (Accepted by AAAI 2023).

## Prerequisites

- Python 3.6
- Pytorch 1.0
- CUDA 10.1

## Get Started

The Deformable ConvNets V2 (DCNv2) module in our code adopts  [chengdazhi's implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0).

You can compile the code according to your machine. 

```shell
sh make.sh
```

Please make sure your machine has a GPU, which is required for the DCNv2 module.

## Evaluation

- Download testsets ([CBSD68, Kodak, McMaster](https://github.com/cszn/FFDNet/tree/master/testsets)), run

`python test.py -- test_noiseL 30|50|70 --test_data Kodak24|BSD68|MCMaster`



