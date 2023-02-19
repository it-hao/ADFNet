import argparse
import glob
import os
import torch
import cv2
import numpy as np
import time 
import torch.nn as nn
from option import args
from torch.autograd import Variable
from utils import logger, batch_PSNR_SSIM_v1, forward_chop
from skimage import img_as_ubyte
from model.adfnet import Net

torch.manual_seed(args.seed)
parser = argparse.ArgumentParser(description="AFDNet")
parser.add_argument("--model", type=str, default='afdnet', help="Mode name")
parser.add_argument('--weights', type=str, default='./checkpoints/adfnet_n50', help="Path to pretrained model")
parser.add_argument("--input_dir", type=str, default='./testsets/Set12', help='test on Set12, BSD68 and Urban100')
parser.add_argument("--result_dir", type=str, default='./res/adfnet', help='Save restoration results')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
parser.add_argument("--rgb_range", type=int, default=1.)
parser.add_argument("--save_images", action='store_true', help='Save denoised images in result directory')
parser.add_argument("--chop", action='store_true')
parser.add_argument("--ensemble", action='store_true')
opt = parser.parse_args()
lg = logger(opt.model, 'res/' + 'adfnet.log')

input_dir_name = os.path.split(opt.input_dir)[-1]

if opt.ensemble:
    save_dir = os.path.join(opt.result_dir + '_n' + str(opt.test_noiseL) + '_plus', input_dir_name) 
else:
    save_dir = os.path.join(opt.result_dir + '_n' + str(opt.test_noiseL), input_dir_name)


def normalize(data, rgb_range):
    return data / (255. / rgb_range)

def main():
    # Build model
    lg.info('Noise level: %s' % (opt.test_noiseL))
    # lg.info('Loading model: %s' % (os.path.split(opt.weights)[-1]))
    lg.info('Loading model: %s Ensemble: %s' % (opt.weights, opt.ensemble))
    net = Net()
    model = Net().to('cuda')
    # model = nn.DataParallel(model).module # 多卡运行
    model.load_state_dict(
                torch.load(
                    os.path.join(opt.weights),
                    **{}
                ),
                strict=False
    )

    model.eval()
    # load data info
    lg.info('Loading data info: %s' % (os.path.split(opt.input_dir)[-1]))
    files_source = glob.glob(os.path.join(opt.input_dir, '*.png'))
    files_source.sort()

    # process data
    psnr_test = 0
    ssim_test = 0
    time_start = time.time()
    with torch.no_grad():
        for f in files_source:
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:, :, 0]), opt.rgb_range)
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)
            torch.manual_seed(args.seed)
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / (255. / opt.rgb_range))

            # noisy image
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            with torch.no_grad():
                if opt.ensemble:
                    Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=args.n_GPUs, ensemble=True), 0., opt.rgb_range)
                else:
                    Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=args.n_GPUs, ensemble=False), 0., opt.rgb_range)

            psnr_score, ssim_score = batch_PSNR_SSIM_v1(Out, ISource) # n c h w
            psnr_test += psnr_score
            ssim_test += ssim_score
            file_name = os.path.split(f)[-1]
            lg.info("%s: PSNR %.4f  SSIM %.4f" % (file_name, psnr_score, ssim_score))

            # save results
            if opt.save_images:
                image = Out.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file_name), img_as_ubyte(image))

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    lg.info("\nPSNR on test data %f, SSIM on test data %f" % (psnr_test, ssim_test))
    lg.info('Finish!\n')
    time_end = time.time()
    print("Tol time:", time_end - time_start)


if __name__ == "__main__":
    main()
