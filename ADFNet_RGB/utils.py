import math
import cv2
import torch
import numpy as np
from skimage import img_as_ubyte
from skimage.measure.simple_metrics import compare_psnr
import logging
import os
import os.path as osp

def logger(name, filepath):
    dir_path = osp.dirname(filepath)
    if not osp.exists(dir_path):
        os.mkdir(dir_path)

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s |[%(lineno)03d]%(filename)-11s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(formatter)
    lg.addHandler(stream_hd)

    file_hd = logging.FileHandler(filepath)
    file_hd.setFormatter(formatter)
    lg.addHandler(file_hd)

    return lg

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# 由于丢失了精度，所以结果会稍微低一点点
def batch_PSNR_SSIM_v1(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    SSIM = 0
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)))
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)))
    return PSNR / Img.shape[0], SSIM / Img.shape[0]


# https://github.com/cszn/DPIR/blob/master/main_dpir_denoising.py
# def batch_PSNR_SSIM_v2(img, imclean):
#     Img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
#     Iclean = imclean.data.squeeze().float().clamp_(0, 1).cpu().numpy()
#     if Img.ndim == 3:
#         Img = np.transpose(Img, (1, 2, 0))
#         Iclean = np.transpose(Iclean, (1, 2, 0))
#     Img = np.uint8((Img*255.0).round())
#     Iclean = np.uint8((Iclean*255.0).round())
#     PSNR = calculate_psnr(Img, Iclean)
#     SSIM = calculate_ssim(Img, Iclean)
#     return PSNR, SSIM


def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def test_x8(model, L):
    E_list = [model(augment_img_tensor(L, mode=i)) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E



def forward_chop(x, nn_model, n_GPUs=1, shave=10, min_size=4000000, ensemble=False):
    scale = 1
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    #############################################
    # adaptive shave
    # corresponding to scaling factor of the downscaling and upscaling modules in the network
    shave_scale = 8
    # max shave size
    shave_size_max = 24
    # get half size of the hight and width
    h_half, w_half = h // 2, w // 2
    # mod
    mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
    # ditermine midsize along height and width directions
    h_size = mod_h * shave_scale + shave_size_max
    w_size = mod_w * shave_scale + shave_size_max
    # h_size, w_size = h_half + shave, w_half + shave
    ###############################################
    # h_size, w_size = adaptive_shave(h, w)
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if not ensemble:
                sr_batch = nn_model(lr_batch)
            else:
                sr_batch = test_x8(nn_model, lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, nn_model, shave=shave, min_size=min_size, ensemble=ensemble) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    # output = Variable(x.data.new(b, c, h, w), volatile=True)
    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

# def batch_PSNR(img, imclean, border=0):
#     Img = img.data.cpu().numpy()
#     Iclean = imclean.data.cpu().numpy()
#     Img = img_as_ubyte(Img)
#     Iclean = img_as_ubyte(Iclean)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
#     return (PSNR/Img.shape[0])

# def batch_SSIM(img, imclean, border=0):
#     Img = img.data.cpu().numpy()
#     Iclean = imclean.data.cpu().numpy()
#     Img = img_as_ubyte(Img)
#     Iclean = img_as_ubyte(Iclean)
#     SSIM = 0
#     for i in range(Img.shape[0]):
#         SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
#     return (SSIM/Img.shape[0])