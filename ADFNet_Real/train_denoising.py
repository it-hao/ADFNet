import os
import torch
import random
import time
import utils
import torch.nn as nn
import torch.optim as optim
import numpy as np

from thop import profile
from torch.utils.data import DataLoader
from dataloaders.data_rgb import get_training_data, get_validation_data
from networks.adfnet import Net
from losses import CharbonnierLoss
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from config import Config

torch.backends.cudnn.benchmark = True
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.cuda.set_device(0) # cuDNN error: CUDNN_STATUS_INTERNAL_ERROR 报错

######### Setting #############
start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION
lg = utils.logger(mode, 'log/' + session + '.log')

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

######### Model ###########
model_restoration = Net()
model_restoration.cuda()

# 模型复杂度
# input = torch.randn(1, 3, 480, 320).cuda()
# flops, params = profile(model_restoration, inputs=(input,))
# lg.info('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    lg.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

new_lr = opt.OPTIM.LR_INITIAL
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)


######### Scheduler ###########
warmup_epochs = 3
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.OPTIM.NEPOCH_DECAY, gamma=0.5)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs+40, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)

    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]

    lg.info('------------------------------------------------------------------------------')
    lg.info("==> Resuming Training with learning rate: {}".format(new_lr))
    lg.info('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoaders ###########
img_options_train = {'patch_size': opt.TRAINING.TRAIN_PS}

train_dataset = get_training_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False)

val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

lg.info('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
lg.info('===> Loading datasets')

best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader) // 3
mixup = utils.MixUp_AUG()
lg.info('Evaluation after every {} Iterations !!!'.format(eval_now))

########## Training #############
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 40 + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 1):
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = mixup.aug(target, input_)

        restored = torch.clamp(model_restoration(input_), 0., 1.)
        # restored = torch.clamp(utils.forward_chop(x=input_, nn_model=model_restoration, ensemble=False), 0., 1.)

        loss = criterion(restored, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0 and epoch > 5:
            if save_images:
                utils.mkdir(result_dir)
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate(tqdm(val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]

                    # restored = torch.clamp(utils.forward_chop(x=input_, nn_model=model_restoration, ensemble=False), 0., 1.)
                    restored = torch.clamp(model_restoration(input_), 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                    if save_images:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                        for batch in range(input_.shape[0]):
                            temp = np.concatenate((input_[batch] * 255, restored[batch] * 255, target[batch] * 255),
                                                  axis=1)
                            utils.save_img(os.path.join(result_dir, filenames[batch][:-4] + '.png'),
                                           temp.astype(np.uint8))

                psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))
                lg.info(
                    "\n[Epoch %d iter %d\t PSNR SIDD: %.4f\t] ----  [best_Epoch_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))

            model_restoration.train()

    scheduler.step()

    lg.info("------------------------------------------------------------------")
    lg.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss, scheduler.get_lr()[0]))
    lg.info("------------------------------------------------------------------\n")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

