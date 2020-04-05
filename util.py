import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, gridspec
import os
import glob
from time import strftime, localtime
from shutil import copy
from scipy.misc import imresize
import torch
import torchvision
from skimage import io, transform
from torchvision import transforms as transforms


def read_data(conf):
    input_images = [read_shave_tensorize(path, conf.must_divide) for path in conf.input_image_path]
    return input_images

def read_data_reg(path):
    input_images = read_shave_tensorize(path, 1)
    return input_images


def read_shave_tensorize(path, must_divide):
    input_np = (np.array(Image.open(path).convert('RGB')) / 255.0)

    input_np_shaved = input_np[:(input_np.shape[0] // must_divide) * must_divide,
                               :(input_np.shape[1] // must_divide) * must_divide,
                               :]

    input_tensor = im2tensor(input_np_shaved)

    return input_tensor


def tensor2im(image_tensors, imtype=np.uint8):

    if not isinstance(image_tensors, list):
        image_tensors = [image_tensors]

    image_numpys = []
    for image_tensor in image_tensors:
        # Note that tensors are shifted to be in [-1,1]
        image_numpy = image_tensor.detach().cpu().float().numpy()

        if np.ndim(image_numpy) == 4:
            image_numpy = image_numpy.transpose((0, 2, 3, 1))

        image_numpy = np.round((image_numpy.squeeze(0) + 1) / 2.0 * 255.0)
        image_numpys.append(image_numpy.astype(imtype))

    if len(image_numpys) == 1:
        image_numpys = image_numpys[0]

    return image_numpys


def im2tensor(image_numpy, int_flag=False):
    # the int flag indicates whether the input image is integer (and [0,255]) or float ([0,1])
    if int_flag:
        image_numpy /= 255.0
    # Undo the tensor shifting (see tensor2im function)
    transformed_image = np.transpose(image_numpy, (2, 0, 1)) * 2.0 - 1.0
    return torch.FloatTensor(transformed_image).unsqueeze(0).cuda()


def random_size(orig_size, curriculum=True, i=None, iter_for_max_range=None, must_divide=8.0,
                min_scale=0.25, max_scale=2.0, max_transform_magniutude=0.3):
    cur_max_scale = 1.0 + (max_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else max_scale
    cur_min_scale = 1.0 + (min_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else min_scale
    cur_max_transform_magnitude = (max_transform_magniutude * np.clip(1.0 * i / iter_for_max_range, 0, 1)
                                   if curriculum else max_transform_magniutude)

    # set random transformation magnitude. scalar = affine, pair = homography.
    random_affine = -cur_max_transform_magnitude + 2 * cur_max_transform_magnitude * np.random.rand(2)

    # set new size for the output image
    new_size = np.array(orig_size) * (cur_min_scale + (cur_max_scale - cur_min_scale) * np.random.rand(2))

    return tuple(np.uint32(np.ceil(new_size * 1.0 / must_divide) * must_divide)), random_affine


def image_concat(g_preds, d_preds=None, size=None):
    hsize = g_preds[0].shape[0] + 6 if size is None else size[0]
    results = []
    if d_preds is None:
        d_preds = [None] * len(g_preds)
    for g_pred, d_pred in zip(g_preds, d_preds):
        # noinspection PyUnresolvedReferences
        dsize = g_pred.shape[1] if size is None or size[1] is None else size[1]
        result = np.ones([(1 + (d_pred is not None)) * hsize, dsize, 3]) * 255
        if d_pred is not None:
            # d_pred_new = imresize((np.concatenate([d_pred] * 3, 2) - 128) * 2, (int(result.shape[0]/4), int(result.shape[1])), interp='nearest')
            # g_pred_new = imresize(g_pred, (int(result.shape[0]/4), int(result.shape[1])), interp='nearest')

            g_pred_new = g_pred
            d_pred_new = imresize((np.concatenate([d_pred] * 3, 2) - 128) * 2, g_pred.shape[0:2], interp='nearest')

            tempim = np.concatenate([g_pred_new, d_pred_new], 0)
            # tempim2 = result[hsize-g_pred.shape[0]:hsize+g_pred.shape[0], :g_pred.shape[1], :]
            # tempim = resize_tensor(tempim, tempim2.shape[0], tempim2.shape[1])
            result[hsize-g_pred_new.shape[0]:hsize+g_pred_new.shape[0], :g_pred_new.shape[1], :] = tempim
        else:
            result[hsize - g_pred.shape[0]:, :, :] = g_pred
        results.append(np.uint8(np.round(result)))

    return np.concatenate(results, 1)


def save_image(image_tensor, image_path):
    image_pil = Image.fromarray(tensor2im(image_tensor), 'RGB')
    image_pil.save(image_path)


def get_scale_weights(i, max_i, start_factor, input_shape, min_size, num_scales_limit, scale_factor):
    num_scales = np.min([np.int(np.ceil(np.log(np.min(input_shape) * 1.0 / min_size)
                                        / np.log(scale_factor))), num_scales_limit])

    # if i > max_i * 2:
    #     i = max_i * 2
    num_scales = num_scales_limit

    factor = start_factor ** ((max_i - i) * 1.0 / max_i)

    un_normed_weights = factor ** np.arange(num_scales)
    weights = un_normed_weights / np.sum(un_normed_weights)
    #
    # np.clip(i, 0, max_i)
    #
    # un_normed_weights = np.exp(-((np.arange(num_scales) - (max_i - i) * num_scales * 1.0 / max_i) ** 2) / (2 * sigma ** 2))
    # weights = un_normed_weights / np.sum(un_normed_weights)

    return weights


class Visualizer:
    def __init__(self, conf):
        self.conf = conf
        self.G_loss = [None] * conf.max_iters
        self.D_loss = [None] * conf.max_iters
        self.D_loss_real = [None] * conf.max_iters
        self.D_loss_fake = [None] * conf.max_iters


        if conf.reconstruct_loss_stop_iter > 0:
            self.Rec_loss = [None] * conf.max_iters

    def recreate_fig(self):
        self.fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(8, 8)
        self.result = self.fig.add_subplot(gs[0:8, 0:4])
        self.gan_loss = self.fig.add_subplot(gs[0:2, 5:8])
        self.reconstruct_loss = self.fig.add_subplot(gs[3:5, 5:8])
        self.reconstruction = self.fig.add_subplot(gs[7, 6])
        self.real_example = self.fig.add_subplot(gs[7, 5])
        self.d_map_real = self.fig.add_subplot(gs[7, 7])

        # First plot data
        self.plot_gan_loss = self.gan_loss.plot([], [], 'b-',
                                                [], [], 'c--',
                                                [], [], 'g--',
                                                [], [], 'r--')
        self.gan_loss.legend(('Generator loss', 'Discriminator loss', 'Discriminator loss (real image)', 'Discriminator loss (fake image)'))
        self.gan_loss.set_ylim(0, 1)

        if self.conf.reconstruct_loss_stop_iter > 0:
            self.plot_reconstruct_loss = self.reconstruct_loss.semilogy([], [])

        # Set titles
        self.gan_loss.set_title('Gan Losses')
        self.reconstruct_loss.set_title('Reconstruction Loss')
        self.reconstruction.set_title('Reconstruction')
        self.d_map_real.set_xlabel('Current Discriminator \n map for real example')
        self.real_example.set_xlabel('Real example')
        self.result.set_title('Current result')

        self.result.axes.get_xaxis().set_visible(False)
        self.result.axes.get_yaxis().set_visible(False)
        self.reconstruction.axes.get_xaxis().set_visible(False)
        self.reconstruction.axes.get_yaxis().set_visible(False)
        self.d_map_real.axes.get_yaxis().set_visible(False)
        self.real_example.axes.get_yaxis().set_visible(False)
        self.result.axes.get_yaxis().set_visible(False)

    def test_and_display(self, p, i, losses_g, losses_d, losses_d_real, losses_d_fake, losses_reconst, lr_g, input_tensor, real_example, g_pred, d_pred_fake, d_pred_real, reconst, scale_weights, discriminator, level, client_bot, bot):
        if not i % self.conf.print_freq and i > 0:
            self.G_loss[i-self.conf.print_freq:i] = losses_g.detach().cpu().float().numpy().tolist()
            self.D_loss[i-self.conf.print_freq:i] = losses_d.detach().cpu().float().numpy().tolist()
            self.D_loss_real[i - self.conf.print_freq:i] = losses_d_real.detach().cpu().float().numpy().tolist()
            self.D_loss_fake[i-self.conf.print_freq:i] = losses_d_fake.detach().cpu().float().numpy().tolist()
            if self.conf.reconstruct_loss_stop_iter > i:
                self.Rec_loss[i-self.conf.print_freq:i] = losses_reconst.detach().cpu().float().numpy().tolist()

            # if self.conf.reconstruct_loss_stop_iter < i:
            #     print('iter: %d, G_loss: %f, D_loss_real: %f, D_loss_fake: %f, LR: %f' %
            #           (i, self.G_loss[i-1], self.D_loss_real[i-1], self.D_loss_fake[i-1],
            #            lr_g))
            # else:
            #     print('iter: %d, G_loss: %f, D_loss_real: %f, D_loss_fake: %f, Rec_loss: %f, LR: %f' %
            #           (i, self.G_loss[i-1], self.D_loss_real[i-1], self.D_loss_fake[i-1], self.Rec_loss[i-1],
            #            lr_g))

        if not i % self.conf.display_freq and i > 0:
            plt.gcf().clear()
            plt.close()
            self.recreate_fig()



            g_preds = [real_example, g_pred]
            d_preds = [discriminator.forward(real_example.detach(), scale_weights, p, level),
                       d_pred_fake]
            reconstructs = reconst
            input_size = g_pred.shape[2:]

            result = image_concat(tensor2im(g_preds), tensor2im(d_preds), (input_size[0]*2, input_size[1]*2))
            self.plot_gan_loss[0].set_data(range(i), self.G_loss[:i])
            self.plot_gan_loss[1].set_data(range(i), self.D_loss[:i])
            # self.plot_gan_loss[2].set_data(range(i), self.D_loss_real[:i])
            # self.plot_gan_loss[3].set_data(range(i), self.D_loss_fake[:i])
            self.gan_loss.set_xlim(0, i)

            if self.conf.reconstruct_loss_stop_iter > i:
                self.plot_reconstruct_loss[0].set_data(range(i), self.Rec_loss[:i])
                self.reconstruct_loss.set_ylim(np.min(self.Rec_loss[:i]), np.max(self.Rec_loss[:i]))
                self.reconstruct_loss.set_xlim(0, i)

            self.result.imshow(np.clip(result, 0, 255), vmin=0, vmax=255)
            self.real_example.imshow(np.clip(tensor2im(real_example[0:1, :, :, :]), 0, 255), vmin=0, vmax=255)
            self.d_map_real.imshow(d_pred_real[0:1, :, :, :].detach().cpu().float().numpy().squeeze(),
                                   cmap='gray', vmin=0, vmax=1)
            if self.conf.reconstruct_loss_stop_iter > i:
                self.reconstruction.imshow(np.clip(image_concat([tensor2im(reconstructs)]), 0, 255), vmin=0, vmax=255)

            plt.savefig(self.conf.output_dir_path + '/monitor_%d' % i)

            im = read_data_reg(self.conf.output_dir_path + '/monitor_%d.png' % i)
            try:
                if bot:
                    client_bot.upload_tensor_image(im, self.conf.ch_name)
            except:
                pass

            save_image(g_pred, self.conf.output_dir_path + '/result_iter_%d.png' % i)

            try:
                im = read_data_reg(self.conf.output_dir_path + '/result_iter_%d.png' % i)
                if bot:
                    client_bot.upload_tensor_image(im, self.conf.ch_name)
            except:
                pass


def prepare_result_dir(conf):
    # Create results directory
    conf.direct_name = conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
    conf.output_dir_path += '/' + conf.direct_name
    os.makedirs(conf.output_dir_path)

    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if conf.create_code_copy:
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.output_dir_path)
        if conf.resume:
            copy(conf.resume, os.path.join(conf.output_dir_path, 'starting_checkpoint.pth.tar'))
    return conf.output_dir_path


def homography_based_on_top_corners_x_shift(rand_h):
    if(rand_h[0]>1e-4 and rand_h[1]>1e-4):
        rand_h=rand_h* (1e-3)
    p = np.array([[1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
                  [0, 0, 0, 1., 1., -1., 1., 1., -1.],
                  [-1., -1., -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
                  [0, 0, 0, -1, -1, -1, 1, 1, 1],
                  [1, 0, -1, 0, 0, 0, 1, 0, -1],
                  [0, 0, 0, 1, 0, -1, 0, 0, 0],
                  [-1, 0, -1, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, -1, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3).cuda()


def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    a = 1
    b = 1
    y, x = torch.meshgrid((torch.linspace(-b, b, np.int(size[-2]*a)), torch.linspace(-b, b, np.int(size[-1]*a))))
    n = np.int(size[-2] * a) * np.int(size[-1] * a)
    hxy = torch.ones(n, 3, dtype=torch.float)
    hxy[:, 0] = x.contiguous().view(-1)
    hxy[:, 1] = y.contiguous().view(-1)
    out = hxy[None, ...].cuda().matmul(theta.transpose(1, 2))
    # normalize
    out = out[:, :, :2] / out[:, :, 2:]
    return out.view(theta.shape[0], np.int(size[-2]*a), np.int(size[-1]*a), 2)



def resize_tensor(input_tens, h, w):
    final_output = None
    input_tensors = input_tens
    if input_tens.shape[2] == h and input_tens.shape[3] == w:
        return input_tens
    # save_image(input_tensors, "/home/shahafw/Desktop/PinGAN/try1/res1.png")
    batch_size, channel, height, width = input_tensors.shape
    input_tensors = torch.squeeze(input_tensors, 1)

    # image = io.imread("/home/shahafw/Desktop/PinGAN/bull.png")
    temp1 = tensor2im(input_tensors)
    input_tensors = transform.resize(temp1, (h, w),mode ="reflect", anti_aliasing=True)

    # for img in input_tensors:
    #     newer = torchvision.transforms.Resize(img, (h, w))

        # img_PIL = transforms.ToPILImage()(img.cpu())
        # img_PIL = torchvision.transforms.Resize([h, w])(img_PIL)
        # img_PIL = torchvision.transforms.ToTensor()(img_PIL)
        # if final_output is None:
        #     final_output = img_PIL
        # else:
        #     final_output = torch.cat((final_output, img_PIL), 0)
    # final_output = torch.unsqueeze(final_output, 1)
    final_output = im2tensor(input_tensors).cuda()
    return final_output

def save_image(image_tensor, image_path):
    image_pil = Image.fromarray(tensor2im(image_tensor), 'RGB')
    image_pil.save(image_path)


def parFreeze(model, cond):
    for par in model.parameters():
        par.requires_grad = cond