import argparse
import torch.nn as nn
import os
import torch
import numpy as np
import utils
import scipy.io as sio

from dataloaders.data_rgb import get_validation_data
from networks.adfnet import Net
from torch.utils.data import DataLoader
from skimage import img_as_ubyte
from tqdm import tqdm

'''
    从文件夹中读取数据集，测试后得到相应的去噪后的图像，得到提交到网站的测试文件 SubmitSrgb.mat
    测试官网===> http://130.63.97.225/sidd/benchmark_submit.php
'''
parser = argparse.ArgumentParser(description='RGB denoising evaluation on SIDD val dataset')
parser.add_argument('--input_dir', default='./testsets/sidd/val', type=str)
parser.add_argument('--results_dir', default='./results/sidd/val', type=str)
parser.add_argument("--weights", default="./checkpoints/adfnet.pth", type=str, help='path of pre_trained file')
parser.add_argument('--gpus', default='0,1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=8, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--ensemble', action='store_true', help='if or not using self_ensemble strategy')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def main():
    results_png_dir = args.results_dir  + '/png/'
    results_mat_dir = args.results_dir  + '/mat/'
    utils.mkdir(results_png_dir)
    utils.mkdir(results_mat_dir)

    test_dataset = get_validation_data(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, num_workers=8)

    model = Net()
    print("===>Testing using weights: ", args.weights)
    # model = nn.DataParallel(model).cuda()
    utils.load_checkpoint(model, args.weights)
    model.cuda()

    model.eval()
    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        Idenoised = np.empty((40, 32), dtype=object)
        i = 0
        j = 0
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt = data_test[0].cuda()
            rgb_noisy = data_test[1].cuda()
            filenames = data_test[2]
            if args.ensemble:
                rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model, ensemble=True), 0., 1.)
            else:
                rgb_restored = torch.clamp(model(rgb_noisy) , 0., 1.)
                # rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model, ensemble=False), 0., 1.)

            # tensor=====>numpy
            rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            # ============================== PSNR, SSIM Value ===========================
            psnr, ssim = utils.batch_PSNR_SSIM(rgb_restored, rgb_gt)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            # ============================== PSNR, SSIM Value ===========================

            # save images
            if args.save_images:
                for batch in range(len(rgb_gt)):
                    denoised_img = img_as_ubyte(rgb_restored[batch])
                    utils.save_img(results_png_dir + filenames[batch][:-4] + '.png', denoised_img)

                    if j == 32:
                        i += 1
                        j = 0

                    Idenoised[i, j] = denoised_img
                    j += 1

    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)
    print("PSNR = %.4f, SSIM =  %.4f" % (psnr_val_rgb, ssim_val_rgb))

    submit_data = {
        'DenoisedBlocksSrgb': Idenoised
    }
    sio.savemat(
        os.path.join(results_mat_dir, 'SubmitSrgb.mat'),
        submit_data
    )

if __name__ == "__main__":
    main()
