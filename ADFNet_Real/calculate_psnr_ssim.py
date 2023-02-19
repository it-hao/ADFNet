import skimage.io as io
import numpy as np
import math
import cv2
import os
import glob
import argparse
from skimage.measure.simple_metrics import compare_psnr

parser = argparse.ArgumentParser(description="Calculate PSNR and SSIM")
parser.add_argument('--source_dir', type=str, default='../../../data_noise/SIDD/test/hq',
                    help="Ground Truth Dircetory")
parser.add_argument("--test_dir", type=str, default='./results/denoising/sidd/',
                    help='Test Image Directory')
parser.add_argument('--source_path', type=str, default='../../../data_noise/SIDD/test/hq/0000-0000.png',
                    help="The Path of Single Ground Truth Image")
parser.add_argument("--test_path", type=str, default='../../../data_noise/SIDD/test/hq/0000-0000.png',
                    help='The Path of Test Image Directory')
opt = parser.parse_args()

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

################## calc_metrics ######################
def calc_metrics(img1, img2, crop_border, test_Y=True):
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:
        if img1.shape[2] == 3:
            im1_in = rgb2ycbcr(img1)
            im2_in = rgb2ycbcr(img2)
            im1_in = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
            im2_in = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 3.'.format(img1.ndim))
    else:
        im1_in = img1
        im2_in = img2

    psnr_value = calc_psnr(im1_in * 255, im2_in * 255)
    # psnr_value = compare_psnr(im1_in, im2_in, data_range=1)
    ssim_value = calc_ssim(im1_in * 255, im2_in * 255)
    return psnr_value, ssim_value


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def multi_test():
    hr_dir = opt.source_dir
    sr_dir = opt.test_dir

    psnr_sum = 0
    ssim_sum = 0

    hr_list = sorted(
        glob.glob(os.path.join(hr_dir, '*.png'))
    )
    sr_list = sorted(
        glob.glob(os.path.join(sr_dir, '*.png'))
    )

    for i in range(len(hr_list)):
        filename = os.path.basename(hr_list[i])
        hr_img = io.imread(hr_list[i])
        sr_img = io.imread(sr_list[i])
        psnr_, ssim_ = calc_metrics(hr_img, sr_img, 0, False)
        psnr_sum += psnr_
        ssim_sum += ssim_
        print(filename, ":", psnr_, ssim_)

    print("Tol: psnr=[", psnr_sum / len(hr_list), "] ssim=[", ssim_sum / len(hr_list), "]")
    print("Finished!")


def single_test():
    hr_path = opt.source_path
    sr_path = opt.test_path
    hr_img = io.imread(hr_path)
    sr_img = io.imread(sr_path)
    psnr_, ssim_ = calc_metrics(hr_img, sr_img, 0, False)
    print("Single: psnr=[", psnr_, "] ssim=[", ssim_, "]")
    print("Finished!")


if __name__ == '__main__':
    multi_test()
