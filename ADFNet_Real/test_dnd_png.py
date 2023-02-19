import numpy as np
import os
import argparse
import torch.nn as nn
import torch
import utils
from tqdm import tqdm
import scipy.io as sio
from skimage import img_as_ubyte
from networks.adfnet import Net
from torch.utils.data import DataLoader
from dataloaders.data_rgb import get_test_data
from utils.bundle_submissions import bundle_submissions_srgb_v1

parser = argparse.ArgumentParser(description='RGB denoising evaluation on DND dataset')
parser.add_argument('--input_dir', default='./testsets/dnd', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/dnd', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/adfnet.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
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

    test_dataset = get_test_data(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

    model_restoration = Net()

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()

    model_restoration = nn.DataParallel(model_restoration)

    model_restoration.eval()

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            if args.ensemble:
                rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model_restoration), 0., 1.)
            else:
                # rgb_restored = torch.clamp(model_restoration(rgb_noisy) , 0., 1.)
                rgb_restored = torch.clamp(utils.forward_chop(x=rgb_noisy, nn_model=model_restoration, ensemble=False), 0., 1.)

            rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_images:
                for batch in range(len(rgb_noisy)):
                    denoised_img = img_as_ubyte(rgb_restored[batch])
                    utils.save_img(os.path.join(result_dir_img, filenames[batch][:-4] + '.png'), denoised_img)
                    sio.savemat(os.path.join(result_dir_mat, filenames[batch][:-4] + '.mat'), {'Idenoised_crop': np.float32(rgb_restored[batch])})

    if args.save_images:
        bundle_submissions_srgb_v1(result_dir_mat, 'bundled')
        os.system("rm {}".format(result_dir_mat + '/*.mat'))


if __name__ == '__main__':
    main()
