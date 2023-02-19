import numpy as np
import os
import argparse
import scipy.io as sio
import torch
import torch.nn as nn
import utils

from tqdm import tqdm
from skimage import img_as_ubyte
from networks.adfnet import Net

parser = argparse.ArgumentParser(description='RGB denoising evaluation on SIDD benchmark dataset')
parser.add_argument('--input_dir', default='./testsets/sidd/benchmark', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/sidd/benchmark', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/adfnet.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--ensemble', action='store_true', help='using self_ensemble strategy')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def main():
    if args.save_images:
        result_dir_img = os.path.join(args.result_dir, 'png')
        result_dir_mat = os.path.join(args.result_dir, 'mat')
        utils.mkdir(result_dir_img)
        utils.mkdir(result_dir_mat)

    model_restoration = Net()

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # Process data
    filepath = os.path.join(args.input_dir, 'BenchmarkNoisyBlocksSrgb.mat')
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['BenchmarkNoisyBlocksSrgb']))
    Inoisy /= 255.

    restored = np.empty((40, 32), dtype=object)

    with torch.no_grad():
        for i in tqdm(range(40)):
            for k in range(32):
                rgb_noisy = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                if args.ensemble:
                    rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model_restoration), 0., 1.)
                else:
                    # rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model_restoration, ensemble=False), 0., 1.)
                    rgb_restored = torch.clamp(model_restoration(rgb_noisy), 0., 1.)

                rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().squeeze(0).numpy()

                img = img_as_ubyte(rgb_restored)
                restored[i, k] = img

                if args.save_images:
                    save_file = os.path.join(result_dir_img, '%04d-%04d.png' % (i, k))
                    utils.save_img(save_file, img)

    # save denoised data
    if args.save_images:
        sio.savemat(os.path.join(result_dir_mat, 'SubmitSrgb.mat'), {"DenoisedBlocksSrgb": restored})


if __name__ == '__main__':
    main()
