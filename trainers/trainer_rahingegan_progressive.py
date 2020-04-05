import os, cv2
import copy
import torch
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import *
from util import *
from torch.autograd import Variable
import torch.autograd as autograd
from architectures.architecture_pggan import *
from radam import RAdam
from optimizer import Lookahead
from configs import Config
import slack_weizmann
from skvideo.io import FFmpegWriter


class Trainer_RAHINGEGAN_Progressive():
    def __init__(self, netD, netG, device, train_ds, lr_D = 0.0004, lr_G = 0.0001, drift = 0.001, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = False, conf = None, bot =False):
        self.sz = netG.sz
        self.netD = netD
        self.netG = netG
        self.train_ds = train_ds
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.drift = drift
        self.device = device
        self.resample = resample
        self.Ganloss = GANLoss()
        self.Reconstloss = WeightedMSELoss()
        self.Level1Loss = WeightedMSELoss()
        self.vis = Visualizer(conf)
        self.bot =bot

        #optimization steps
        self.optimizerD = RAdam(self.netD.parameters(), lr=self.lr_D, betas = (0, 0.99))
        self.optimizerD = Lookahead(base_optimizer=self.optimizerD, k=5, alpha=0.5)
        self.optimizerG = RAdam(self.netG.parameters(), lr=self.lr_G, betas = (0, 0.99))
        self.optimizerG = Lookahead(base_optimizer=self.optimizerG, k=5, alpha=0.5)
        self.real_label = 1
        self.fake_label = 0
        self.nz = self.netG.nz

        #Bot for slack to send updates
        self.clientbot = slack_weizmann.slack_weizmann(conf)


        self.fixed_noise = generate_noise(49, self.nz, self.device)
        self.loss_interval = loss_interval
        self.image_interval = image_interval
        self.snapshot_interval = snapshot_interval

        self.errD_records = []
        self.errG_records = []
        self.stage = 0
        self.input_im = None
        self.real_im = None
        self.conf = conf

        # Keeping track of losses- prepare tensors
        self.losses_D_gan = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_G_gan = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_real = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_fake = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_G_reconstruct = torch.FloatTensor(conf.print_freq).cuda()
        if self.conf.reconstruct_loss_stop_iter > 0:
            self.losses_D_reconstruct = torch.FloatTensor(conf.print_freq).cuda()

        self.save_cnt = 0
        self.save_img_dir = save_img_dir
        self.save_snapshot_dir = save_snapshot_dir
        self.geo = GeoTransform()
        self.cur_iter= 0
        if(not os.path.exists(self.save_img_dir)):
            os.makedirs(self.save_img_dir)
        if(not os.path.exists(self.save_snapshot_dir)):
            os.makedirs(self.save_snapshot_dir)

    def train(self, res_num_epochs, res_percentage, bs):
        p = 0
        last_p = 0

        res_percentage = [None] + res_percentage
        largest_factor = self.conf.stage_factor ** (self.conf.stage_size)

        #base input image
        self.input_im = resize_tensor(self.train_ds, int(self.train_ds.shape[2]/largest_factor), int(self.train_ds.shape[3]/largest_factor))

        #create ch for slack in case you choose to
        ch_name = self.conf.direct_name
        self.conf.ch_name = ch_name.replace("_", "").lower()
        if self.bot:
            self.clientbot.create_public_ch(self.conf.ch_name)



        for i, (num_epoch, percentage, cur_bs) in enumerate(zip(res_num_epochs, res_percentage, bs)):
            train_dl = self.train_ds
            train_dl_len = 2000
            if(percentage is None):
                num_epoch_transition = 0
            else:
                num_epoch_transition = int(num_epoch * percentage)

            cnt = 1
            counterSet = 0
            for epoch in range(num_epoch):
                p = i
                if(self.resample):
                    train_dl_iter = iter(train_dl)
                for j in range(train_dl_len):
                    self.cur_iter+=1
                    if(epoch < num_epoch_transition):
                        p = i + cnt / (train_dl_len * num_epoch_transition) - 1
                        cnt += 1

                    #update stage to next one
                    if last_p == int(last_p) and (p != int(p) or p >= last_p+1):
                        self.stage += 1
                        self.lr_D /= 2.11
                        self.lr_G /= 2.11

                    counterSet += 1

                    stage_factor = self.conf.stage_factor ** (self.conf.stage_size - self.stage)
                    reconst_factor = self.conf.stage_factor ** ( self.stage)


                    #output scale image
                    self.real_im = resize_tensor(self.train_ds, int(self.train_ds.shape[2] / stage_factor),int(self.train_ds.shape[3] / stage_factor))




                    # Determine output size of G (dynamic change) in output scale
                    output_size, random_affine = random_size(orig_size=self.real_im.shape[2:],
                                                             curriculum=self.conf.curriculum,
                                                             i=self.cur_iter,
                                                             iter_for_max_range=self.conf.iter_for_max_range,
                                                             must_divide=self.conf.must_divide,
                                                             min_scale=self.conf.min_scale,
                                                             max_scale=self.conf.max_scale,
                                                             max_transform_magniutude=self.conf.max_transform_magnitude)
                    output_size_start = (int(output_size[0]/reconst_factor), int(output_size[1]/reconst_factor))
                    # data = self.geo.forward(self.train_ds, output_size, random_affine)
                    # (1) : minimizes mean(max(0, 1-(D(x)-mean(D(G(z)))))) + mean(max(0, 1+(D(G(z))-mean(D(x)))))
                    self.netG.zero_grad()
                    self.netD.zero_grad()
                    # real_images = data.to(self.device)
                    bs = 1
                    input_tensor_noised = self.input_im + (torch.rand_like(self.input_im) - 0.5) * 2.0 / 255

                    fake_images = self.netG(input_tensor_noised, p, output_size_start, output_size, random_affine)
                    # calculate the discriminator results for both real & fake
                    scale_weights = get_scale_weights(i=self.cur_iter,
                                                           max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                                           start_factor=self.conf.D_scale_weights_sigma,
                                                           input_shape=fake_images.shape[2:],
                                                           min_size=self.conf.D_min_input_size,
                                                           num_scales_limit=self.conf.D_max_num_scales,
                                                           scale_factor=self.conf.D_scale_factor)
                    # scale_weights = scale_weights[::-1]
                    if self.cur_iter == 401:
                        print(self.cur_iter)
                    c_xr = self.netD(self.real_im, scale_weights, p, self.stage)				# (bs, 1, 1, 1)
                    # c_xr = c_xr.view(-1)						# (bs)
                    c_xf = self.netD(fake_images, scale_weights, p, self.stage)		# (bs, 1, 1, 1)
                    error_d_fake = self.Ganloss(c_xf, False)
                    error_d_real = self.Ganloss(c_xr, True)
                    # c_xf = c_xf.view(-1)						# (bs)
                    # calculate the Discriminator loss
                    errD = (torch.mean(torch.nn.ReLU()(1-(c_xr-torch.mean(c_xf)))) + torch.mean(torch.nn.ReLU()(1+(c_xf-torch.mean(c_xr))))) / 2.0 + self.drift * torch.mean(c_xr ** 2)

                    errD.backward(retain_graph=True)
                    # update D using the gradients calculated previously
                    self.optimizerD.step()


                    # (2) : minimizes mean(max(0, 1-(D(G(z))-mean(D(x))))) + mean(max(0, 1+(D(x)-mean(D(G(z))))))
                    self.netG.zero_grad()
                    if(self.resample):
                        real_images = next(train_dl_iter)[0].to(self.device)
                        noise = generate_noise(bs, self.nz, self.device)
                        fake_images = self.netG(noise, p)

                    # we updated the discriminator once, therefore recalculate c_xr, c_xf
                    c_xr = self.netD(self.real_im, scale_weights, p, self.stage)						# (bs, 1, 1, 1)
                    c_xf = self.netD(fake_images, scale_weights, p, self.stage)			# (bs, 1, 1, 1)
                    # calculate the Generator loss
                    reconst_input = resize_tensor(fake_images, output_size_start[0], output_size_start[1])

                    reconst_image = self.netG(reconst_input, p, self.input_im.shape[2:], self.real_im.shape[2:],random_affine, True, self.stage)
                    # if counterSet > 2000 :
                    if self.stage != 0 :
                        lossG_reg = self.Ganloss(c_xf, True)
                    else:
                        target_output = resize_tensor(self.train_ds, output_size[0], output_size[1])
                        lossG_reg = self.Level1Loss(fake_images, target_output)

                    loss_reconst = self.Reconstloss(reconst_image, self.real_im)
                    errG = lossG_reg + 0.1 * loss_reconst
                    errG.backward()

                    # update G using the gradients calculated previously
                    self.optimizerG.step()

                    self.errD_records.append(float(errD))
                    self.errG_records.append(float(errG))

                    # Accumulate stats
                    # Accumulating as cuda tensors is much more efficient than passing info from GPU to CPU at every iteration
                    self.losses_G_gan[self.cur_iter % self.conf.print_freq] = errG.item()
                    self.losses_D_fake[self.cur_iter % self.conf.print_freq] = error_d_fake.item()
                    self.losses_D_real[self.cur_iter % self.conf.print_freq] = error_d_real.item()
                    self.losses_D_gan[self.cur_iter % self.conf.print_freq] = errD.item()
                    if self.conf.reconstruct_loss_stop_iter > self.cur_iter:
                        self.losses_G_reconstruct[self.cur_iter % self.conf.print_freq] = loss_reconst.item()

                    last_p = p
                    self.p = p




                    if(j % self.loss_interval == 0):

                        message = '[%d/%d] [%d/%d], curr iter: %d errD : %.4f, errG : %.4f, lrD : %.6f, lrG: %.6f' % (
                        epoch + 1, num_epoch, j + 1, train_dl_len, self.cur_iter, errD, errG, self.lr_D, self.lr_G)
                        print(message)
                        if (j % (self.loss_interval * 4) == 0):
                            try:
                                if self.bot:
                                    prog = self.clientbot.make_bar(((j + 1) / train_dl_len) * 100)
                                    self.clientbot.handle_message(message + "\n" + prog, self.conf.ch_name)
                            except:
                                pass



                    if(self.snapshot_interval is not None):
                        if(j % self.snapshot_interval == 0):
                            stage_int = int(p)
                            if(p == stage_int):
                                res = 2 ** (2+stage_int)
                            else:
                                res = 2 ** (3+stage_int)
                            save(os.path.join(self.save_snapshot_dir, 'Res' + str(res) + '_Epoch' + str(epoch) + '_' + str(j) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)

                    self.vis.test_and_display(p, self.cur_iter, self.losses_G_gan, self.losses_D_gan, self.losses_D_real,
                                              self.losses_D_fake, self.losses_G_reconstruct,
                                              self.lr_G, input_tensor_noised, self.real_im, fake_images, c_xf,
                                              c_xr, reconst_image, scale_weights, self.netD, self.stage, self.clientbot, self.bot)

            vid_scale1, vid_scale2 = define_video_scales()
            #function to create video
            self.retarget_video(self.netG, self.real_im, vid_scale1, vid_scale2, 8, self.conf.output_dir_path, self.p, self.stage, self.cur_iter)


    def gradient_penalty(self, real_image, fake_image, scale_weight, p, stage):
        bs = real_image.size(0)
        alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(0, 1).expand(real_image.size()).to(self.device)
        fake = F.interpolate(fake_image, size=real_image.shape[2:], mode='bilinear')

        interpolation = alpha * real_image + (1 - alpha) * fake

        c_xi = self.netD(interpolation, scale_weight, p, stage)

        gradients = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(bs, -1)
        penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return penalty


    def retarget_video(self, gan, input_tensor, scale_1, scale_2, must_divide, output_dir_path, p, stage, name="1"):

        scales_x = np.dstack((scale_1, scale_2))
        max_scale = np.max(scales_x)
        frame_shape = np.uint32(np.array([input_tensor.shape[2], input_tensor.shape[3]]) * max_scale)
        frame_shape[0] += (frame_shape[0] % 2)
        frame_shape[1] += (frame_shape[1] % 2)
        frames = np.zeros([sum(1 for _ in zip(scale_1, scale_2)), frame_shape[0], frame_shape[1], 3])
        for i, (scale_h, scale_w) in enumerate(zip(scale_1, scale_2)):
            output_image = self.test_one_scale(input_tensor=input_tensor,scale=[scale_h, scale_w], must_divide=must_divide, p=p, stage=stage)
            frames[i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image
        writer = FFmpegWriter(output_dir_path + '/vid{name}.mp4'.format(name=name), verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

        for i, _ in enumerate(zip(scale_1, scale_2)):
            for j in range(3):
                writer.writeFrame(frames[i, :, :, :])
        writer.close()

    def test_one_scale(self, input_tensor, scale, must_divide, affine=None, return_tensor=False,
                       size_instead_scale=False, p=0, stage=0):
        with torch.no_grad():
            in_size = input_tensor.shape[2:]
            if size_instead_scale:
                out_size = scale
            else:
                out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                            np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))

                out_size_start = (int(out_size[0]/(2**stage)),int(out_size[1]/(2**stage)))

            output_tensor = self.test(input_tensor=input_tensor,
                                      output_size_start=out_size_start,
                                       output_size=out_size,
                                       rand_affine=affine,
                                       run_d_pred=False,
                                       run_reconstruct=False,
                                      p=p,
                                      level=stage)
            if return_tensor:
                return output_tensor
            else:
                return tensor2im(output_tensor)

    def test(self, input_tensor, output_size_start, output_size, rand_affine, run_d_pred=True, run_reconstruct=True, p=0, level=0):
        with torch.no_grad():
            self.G_pred = self.netG.forward(Variable(input_tensor.detach()), stage=p, output_size=output_size_start, final_output_size=output_size, shift=[0, 0], reconstruction=False,
                                            level=level)
            if run_d_pred:
                scale_weights_for_output = get_scale_weights(i=self.cur_iter,
                                                             max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                                             start_factor=self.conf.D_scale_weights_sigma,
                                                             input_shape=self.G_pred.shape[2:],
                                                             min_size=self.conf.D_min_input_size,
                                                             num_scales_limit=self.conf.D_max_num_scales,
                                                             scale_factor=self.conf.D_scale_factor)
                scale_weights_for_input = get_scale_weights(i=self.cur_iter,
                                                            max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                                            start_factor=self.conf.D_scale_weights_sigma,
                                                            input_shape=input_tensor.shape[2:],
                                                            min_size=self.conf.D_min_input_size,
                                                            num_scales_limit=self.conf.D_max_num_scales,
                                                            scale_factor=self.conf.D_scale_factor)
                self.D_preds = [self.netD.forward(Variable(input_tensor.detach()), scale_weights_for_input),
                                self.netD.forward(Variable(self.G_pred.detach()), scale_weights_for_output)]
            else:
                self.D_preds = None

            return self.G_pred


class GANLoss(nn.Module):
    """ Receiving the final layer form the discriminator and a boolean indicating whether the input to the
     discriminator is real or fake (generated by generator), this returns a patch"""

    def __init__(self):
        super(GANLoss, self).__init__()

        # Initialize label tensor
        self.label_tensor = None

        # Loss tensor is prepared in network initialization.
        # Note: When activated as a loss between to feature-maps, then a loss-map is created. However, using defaults
        # for BCEloss, this map is averaged and reduced to a single scalar
        self.loss = nn.MSELoss()

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map acording to whether current input to discriminator is real or fake
        self.label_tensor = Variable(torch.ones_like(d_last_layer).cuda(), requires_grad=False) * is_d_input_real

        # Finally return the loss
        return self.loss(d_last_layer, self.label_tensor)
        # return -torch.mean(d_last_layer)

class WeightedMSELoss(nn.Module):
    def __init__(self, use_L1=False):
        super(WeightedMSELoss, self).__init__()

        self.unweighted_loss = nn.L1Loss() if use_L1 else nn.MSELoss()

    def forward(self, input_tensor, target_tensor, loss_mask = None):
        if loss_mask is not None:
            e = (target_tensor.detach() - input_tensor) ** 2
            e *= loss_mask
            return torch.sum(e) / torch.sum(loss_mask)
        else:
            return self.unweighted_loss(input_tensor, target_tensor)


class WGAN(nn.Module):
    def __init__(self):
        super(WGAN,self).__init__()

    def d_loss(self, c_xr, c_xf):
        return torch.mean(c_xf)-torch.mean(c_xr)

    def g_loss(self, x_cf):
        return -torch.mean(x_cf)


def define_video_scales():
    max_v = 1.4
    min_v = 0.5
    max_h = 1.4
    min_h = 0.5
    frames_per_resize = 15
    # max_v = 1.2
    # min_v = 0.8
    # max_h = 1.2
    # min_h = 0.8
    # frames_per_resize = 8

    x = np.concatenate([
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize),
                        np.linspace(min_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, 1, frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize),
                        np.linspace(min_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, 1, frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize)])
    y = np.concatenate([
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, max_h, 2 * frames_per_resize),
                        np.linspace(max_h, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, max_h, 2 * frames_per_resize),
                        np.linspace(max_h, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, min_h, 2 * frames_per_resize)])
    return x, y