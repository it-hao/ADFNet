# Adaptive Dynamic Filtering Network for Image Denoising (AAAI 2023)

> **Abstract:** In image denoising networks, feature scaling is widely used to enlarge the receptive field size and reduce computational costs. This practice, however, also leads to the loss of high-frequency information and fails to consider within-scale characteristics. Recently, dynamic convolution has exhibited powerful capabilities in processing high-frequency information (*e.g.*, edges, corners, textures), but previous works lack sufficient spatial contextual information in filter generation. To alleviate these issues, we propose to employ dynamic convolution to improve the learning of high-frequency and multi-scale features. Specifically, we design a spatially enhanced kernel generation (SEKG) module to improve dynamic convolution, enabling the learning of spatial context information with a very low computational complexity. Based on the SEKG module, we propose a dynamic convolution block (DCB) and a multi-scale dynamic convolution block (MDCB). The former enhances the high-frequency information via dynamic convolution and preserves low-frequency information via skip connections. The latter utilizes shared adaptive dynamic kernels and the idea of dilated convolution to achieve efficient multi-scale feature extraction. The proposed multi-dimension feature integration (MFI) mechanism further fuses the multi-scale features, providing precise and contextually enriched feature representations. Finally, we build an efficient denoising network with the proposed DCB and MDCB, named ADFNet. It achieves better performance with low computational complexity on real-world and synthetic Gaussian noisy datasets.

<details>
  <summary> <strong>Network Architecture</strong> (click to expand) 	</summary>
<table>
  <tr>
    <td> <img src="Figs\fig2.png" alt="Fig2" width="400px"/> </td>
    <td> <img src="Figs\fig3.png" alt="fig3" width="400px"/> </td>
  </tr>
</table> 
<p align="center">
  <img src="Figs\fig4.png" alt="Fig2" width="800px"/>
</p>
</details>

## Environment

- Python 3.6 + Pytorch 1.0 + CUDA 10.1
- numpy
- skimage
- imageio
- cv2

## Get Started

The Deformable ConvNets V2 (DCNv2) module in our code adopts  [chengdazhi's implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0).

You can compile the code according to your machine. 

```shell
sh make.sh
```

Please make sure your machine has a GPU, which is required for the DCNv2 module.

## Training

### RGB image denoising

- Download [DIV2K](https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing) training data (800 training images) to train **ADFNet** or Download [DIV2K](https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing)+[Flickr2K](https://drive.google.com/file/d/1J8xjFCrVzeYccD-LF08H7HiIsmi8l2Wn/view?usp=sharing)+[BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing)+[WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing) to train **ADFNet***.  


- for **ADFNet**

  Run `bash train_adfnet_n10.sh` or `bash train_adfnet_n30.sh`or `bash train_adfnet_n50.sh` or `bash train_adfnet_n70.sh`

- for **ADFNet***

  Run `bash train_adfnet-L_n50.sh`

### Gray image denoising

- Run `bash train_adfnet_n15.sh` or `bash train_adfnet_n25.sh`or `bash train_adfnet_n50.sh` 

### Real-world image denoising

- Download the SIDD-Medium dataset from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

- Generate image patches

  `python generate_patches_SIDD.py --ps 256 --num_patches 300 --num_cores 10`

  and then place all image patches in `./datasets/sidd_patch`

- Download validation images of SIDD and place them in `./testsets/sidd/val`

- Install warmup scheduler

- Train your model with default arguments by running

  Run `bash train_adfnet.sh`

## Evaluation

Part of pre-trained models: [Google drive](https://drive.google.com/file/d/1wYw8mHSyxmutpHTahn_j4wjv_p4sJHeq/view?usp=share_link)  [Baidu cloud](https://pan.baidu.com/s/1eAbY3IBSLigkRJJfoQJ73A&pwd=1995)

### RGB image denoising

- cd ./ADFNet_RGB

- Download models and place it in ./checkpoints

- Download testsets ([CBSD68, Kodak24, McMaster](https://github.com/cszn/FFDNet/tree/master/testsets)) and place it in ./testsets
- Run `python test.py --save_images --chop `

### Gray image denoising

- cd ./ADFNet_Gray

- Download models and place it in ./checkpoints

- Download testsets ([BSD68, Urban100, Set12](https://github.com/cszn/FFDNet/tree/master/testsets)) and place it in ./testsets
- Run `python test.py --save_images --chop `

### Real-world image denoising

- cd ./ADFNet_Real

- Download the model and place it in ./pretrained_models

​	**Testing on SIDD datasets**

- Download sRGB validation [images](https://drive.google.com/drive/folders/1j5ESMU0HJGD-wU6qbEdnt569z7sM3479?usp=sharing) of SIDD and place them in ./datasets/sidd/val

- First, run `test_sidd_val_png.py --save_images --ensemble`, 

  and then run `evaluate_SIDD.m` to calculate the PSNR/SSIM value

- Download benchmark [BenchmarkNoisyBlocksSrgb.mat](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) of SIDD and place them in ./datasets/sidd/benchmark

- Run `test_sidd_benchmark_mat.py --save_images --ensemble`

​	**Testing on DND datasets**

- Download sRGB [images](https://drive.google.com/drive/folders/1-IBw_J0gdlM6AlqSm3Z7XWTXR-So4xzp?usp=sharing) of DND and place them in ./datasets/dnd/
- Run `test_dnd_png.py --save_images --ensemble`

## Citation

If you use ADFNet, please consider citing:

```
@article{shen2022adaptive,
  title={Adaptive Dynamic Filtering Network for Image Denoising},
  author={Shen, Hao and Zhao, Zhong-Qiu and Zhang, Wandi},
  booktitle={AAAI},
  year={2023}
}
```

**Acknowledgment**: This code is based on the [MIRNet](https://github.com/swz30/MIRNet) and [DAGL](https://github.com/jianzhangcs/DAGL) toolbox.
