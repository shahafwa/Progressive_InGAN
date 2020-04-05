import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.architecture_pggan import PGGAN_D, PGGAN_G ,MultiScaleDiscriminator
from trainers.trainer_rahingegan_progressive import Trainer_RAHINGEGAN_Progressive
from utils import save, load
from configs import Config
from util import Visualizer, read_data


# Load configuration
conf = Config().parse()

# Prepare data
input_images = read_data(conf)

dir_name = 'data/celeba'
basic_types = None

lr_D, lr_G = 0.001, 0.001
sz, nc, nz = 128, 3, 3
use_sigmoid = False

data = Dataset(dir_name)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = MultiScaleDiscriminator(conf.output_crop_size,  conf.D_max_num_scales, conf.D_scale_factor, conf.D_base_channels).to(device)
netG = PGGAN_G(sz, nz, nc, conf, True, True).to(device)

trainer = Trainer_RAHINGEGAN_Progressive(netD, netG, device, input_images[0], lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300, conf = conf)


#first argument is number of epochs in each stage , second is the percent of stabilization stage(progressive gan) in each epoch (0.5 means 50% percent).
# trainer.train([3, 8, 12, 16], [0.5, 0.5, 0.5], [16, 16, 16, 16])
trainer.train([5, 8, 9, 12, 16], [0.5, 0.5, 0.5, 0.5], [16, 16, 16, 16, 16])

save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
