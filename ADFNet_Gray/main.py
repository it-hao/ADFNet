import torch
import random
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    return num_params


# if __name__ == '__main__':
######### Set Seeds ###########
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


checkpoint = utility.checkpoint(args)
if checkpoint.ok:
    loader = data.Data(args)

    model = model.Model(args, checkpoint)

    # from thop import profile
    # input = torch.randn(1, 1, 512, 512).cuda()
    # flops, params = profile(model, inputs=(input, 0))
    # print(flops, params)

    # from torchsummaryX import summary
    # summary(model, input, 0)

    # checkpoint.write_log(
    #         'Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9)
    #     )

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

