import torch.nn.functional as F
import argparse
import glob
import os
import torch
import cv2
import numpy as np
from skimage import img_as_ubyte
from torch.autograd import Variable
import torch.nn as nn
from utils import logger, batch_PSNR_SSIM_v1, forward_chop

from model.adfnet import Net

parser = argparse.ArgumentParser(description="Test ADFNet")
parser.add_argument("--model", type=str, default='adfnet', help="Mode name")
parser.add_argument('--logdir', type=str, default='pre_train/color_n50.pt', help="Path to pretrained model") 
parser.add_argument("--test_data", type=str, default='../code/testsets/McMaster', help='test on Kodak24, BSD68, Urban100 and McMaster')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
parser.add_argument("--rgb_range", type=int, default=1.)
parser.add_argument("--save_path", type=str, default='./res/adfnet', help='Save restoration results')
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--chop", type=bool, default=True)
parser.add_argument("--ensemble", type=bool, default=False)
opt = parser.parse_args()
lg = logger(opt.model, 'res/' + 'adfnet.log')

test_data_name = os.path.split(opt.test_data)[-1]

if opt.ensemble:
    save_dir = os.path.join(opt.save_path + '_n' + str(opt.test_noiseL) + '_plus', test_data_name) 
else:
    save_dir = os.path.join(opt.save_path + '_n' + str(opt.test_noiseL), test_data_name)

def normalize(data, rgb_range):
    return data / (255. / rgb_range)

def main():
    # Build model
    lg.info('Noise level: %s' % (opt.test_noiseL))
    # lg.info('Loading model: %s' % (os.path.split(opt.logdir)[-1]))
    lg.info('Loading model: %s Ensemble: %s' % (opt.logdir, opt.ensemble))
    net = Net()
    model = Net().to('cuda')
    
    from thop import profile
    input = torch.randn(1, 3, 480, 320).cuda()
    flops, params = profile(model, inputs=(input,))
    print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))

    model.load_state_dict(
                torch.load(
                    os.path.join(opt.logdir),
                    **{}
                ),
                strict=False
    )

    model.eval()
    
    files_source = glob.glob(os.path.join(opt.test_data, '*'))
    files_source.sort()

    # process data
    psnr_test = 0
    ssim_test = 0
    with torch.no_grad():
        for f in files_source:
            Img = cv2.imread(f, cv2.COLOR_BGR2RGB)
            Img = normalize(np.float32(Img), opt.rgb_range)

            ISource = torch.from_numpy(np.ascontiguousarray(Img)).permute(2, 0, 1).float().unsqueeze(0)
            torch.manual_seed(20)
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / (255. / opt.rgb_range))

            # # noisy image
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

            factor = 8
            h,w = INoisy.shape[2], INoisy.shape[3]  
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            INoisy = F.pad(INoisy, (0,padw,0,padh), 'reflect')

            with torch.no_grad():
                if opt.ensemble:
                    Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=1, ensemble=True), 0., opt.rgb_range)
                else:
                    if opt.chop:
                        Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=1, ensemble=False), 0., opt.rgb_range)
                    else:
                        Out = torch.clamp(model(INoisy), 0., opt.rgb_range)
            
            Out = Out[:, :, 0:h, 0:w]

            psnr_score, ssim_score = batch_PSNR_SSIM_v1(Out, ISource) 
            psnr_test += psnr_score
            ssim_test += ssim_score
            file_name = os.path.split(f)[-1]
            lg.info("%s: PSNR %.4f  SSIM %.4f" % (file_name, psnr_score, ssim_score))

            # save results
            if opt.save:
                image = Out.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, file_name), img_as_ubyte(image))

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    lg.info("\nPSNR on test data %f, SSIM on test data %f" % (psnr_test, ssim_test))
    lg.info('Finish!\n')


if __name__ == "__main__":
    main()
