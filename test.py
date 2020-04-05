import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from architectures.architecture_pggan import GeoTransform
from PIL import Image
import util
from configs import Config
from traceback import print_exc
from skvideo.io import FFmpegWriter
import numpy as np
import torch
import util
import utils
import torch.autograd as autograd
from torch.autograd import Variable
from architectures.architecture_pggan import  PGGAN_G, MultiScaleDiscriminator
from radam import RAdam



def define_video_scales(min_v=0.4, max_v=2.2, min_h=0.4, max_h=2.2, frames_per_resize=10):

    # max_v = 2.2
    # min_v = 0.4
    # max_h = 2.2
    # min_h = 0.4
    # frames_per_resize = 10

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

def retarget_video(gan, input_tensor, scale_1, scale_2, must_divide, output_dir_path, p, stage, name="1"):

    scales_x = np.dstack((scale_1, scale_2))
    max_scale = np.max(scales_x)
    frame_shape = np.uint32(np.array([input_tensor.shape[2] * 2**stage,input_tensor.shape[3] * 2**stage]) * max_scale)
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([sum(1 for _ in zip(scale_1, scale_2)), frame_shape[0], frame_shape[1], 3])
    for i, (scale_h, scale_w) in enumerate(zip(scale_1, scale_2)):
        output_image = test_one_scale(gan=gan, input_tensor=input_tensor,scale=[scale_h, scale_w], must_divide=must_divide, p=p, stage=stage)
        frames[i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image
    writer = FFmpegWriter(output_dir_path + '/vid{name}.mp4'.format(name=name), verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i, _ in enumerate(zip(scale_1, scale_2)):
        for j in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()

def test_one_scale(gan, input_tensor, scale, must_divide, affine=None, return_tensor=False,
                   size_instead_scale=False, p=0, stage=0):
    with torch.no_grad():
        in_size = input_tensor.shape[2:]
        if size_instead_scale:
            out_size = scale
        else:
            out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                        np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))

        output_tensor = test(gan=gan,
                             input_tensor=input_tensor,
                                   output_size=out_size,
                                   rand_affine=affine,
                                   run_d_pred=False,
                                   run_reconstruct=False,
                                  p=p,
                                  level=stage)
        if return_tensor:
            return output_tensor
        else:
            return util.tensor2im(output_tensor)

def test(gan, input_tensor, output_size, rand_affine, run_d_pred=True, run_reconstruct=True, p=0, level=0):
    with torch.no_grad():
        G_pred = gan.forward(Variable(input_tensor.detach()), stage=p, output_size=output_size, shift=[0, 0], reconstruction=False,
                                        level=level)


        return G_pred


def main():

    name_infile_ckptfile_list = [
                                 ['0000', 'fruits.png', '/Fruit_Aug_29_18_07_05'],

    # name_infile_ckptfile_list = [['nrid_%d_075' % i, 'scaled_nird/ours_%d_scaled.jpg' % i, 'ours_%d' % i]
    #                              for i in range(1, 36)]
    ]

    snapshot_iters =[75000]
    n_files = len(name_infile_ckptfile_list)

    for i, (name, input_image_path, test_params_path) in enumerate(name_infile_ckptfile_list):
        for snapshot_iter in snapshot_iters:

            conf = Config().parse(create_dir_flag=False)
            conf.name = 'TEST_' + name #+ '_iter_%dk' % (snapshot_iter / 1000)
            conf.output_dir_path = util.prepare_result_dir(conf)
            conf.input_image_path = [os.path.dirname(os.path.abspath(__file__)) + '/' + input_image_path]
            conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + '/results/' + test_params_path + '/cur_state.state'
            # conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + test_params_path + '/checkpoint_%07d.pth.tar' % snapshot_iter
            # gan = InGAN(conf)
            lr_D, lr_G = 0.001, 0.003
            sz, nc, nz = 128, 3, 3
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            netD = MultiScaleDiscriminator(conf.output_crop_size, conf.D_max_num_scales, conf.D_scale_factor,
                                           conf.D_base_channels).to(device)
            netG = PGGAN_G(sz, nz, nc, conf, True, True).to(device)

            optD = RAdam(netD.parameters(), lr=lr_D, betas=(0, 0.99))
            optG = RAdam(netG.parameters(), lr=lr_G, betas=(0, 0.99))

            try:
                utils.load(conf.test_params_path, netD ,netG, optD, optG)
                input_tensor = util.read_data(conf)
                input_tensor = input_tensor[0]

                largest_factor = conf.stage_factor ** (conf.stage_size)

                input_im = util.resize_tensor(input_tensor, int(input_tensor.shape[2] / largest_factor), int(input_tensor.shape[3] / largest_factor))

                # retarget_video(gan, input_tensor, define_video_scales(), 8, conf.output_dir_path, conf.name)

                # generate_collage_and_outputs(conf, gan, input_tensor)

                vid_scale1, vid_scale2 = define_video_scales(min_v=0.6, max_v=2.2, min_h=0.6, max_h=2, frames_per_resize=12)

                retarget_video(netG, input_im, vid_scale1, vid_scale2, 8, conf.output_dir_path, conf.stage_size, conf.stage_size, 1)

                print('Done with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter ))

            except KeyboardInterrupt:
                raise
            except Exception as e:
                # print 'Something went wrong with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)
                print_exc()


if __name__ == '__main__':
    main()




