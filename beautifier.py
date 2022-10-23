# CycleGAN implementation from Jun-Yan Zhu  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import argparse
import urllib.request
import os
import sys
import torch

import models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataroot', default='/content/cyclegan_dataset', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
# parser.add_argument('--checkpoints_dir', type=str, default=cyclegan_models_dir, help='models are saved here')
# model parameters
parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
# dataset parameters
parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
# parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
# parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
# parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
# parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
# additional parameters
parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
parser.add_argument('--display_freq', type=int, default=800, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
# network saving and loading parameters
parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')


opt, _ = parser.parse_known_args()
opt.isTrain = False
from models import create_model
# from data import create_dataset

str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])
    
opt.epoch = 'beautiful' 

cyclegan_model = create_model(opt)

opt.ngf, opt.ndf = 64, 64
for name in cyclegan_model.model_names:
    if isinstance(name, str):
        save_filename = '%s_net_%s.pth' % (opt.epoch, name)
        if not os.path.exists('pretrained'): os.mkdir('pretrained')
        save_path = 'pretrained/cyclegan_generator.pth'
        if name == 'G_A':
            net = getattr(cyclegan_model, 'net' + name)
            if not os.path.exists(save_path):
                #!wget -O cyclegan_generator.pth https://dl.boxcloud.com/d/1/b1!NNM62Iu7CAUniqipvBqxuFBwgzqaXUC50EIiSclFKWmfIgKp-LtxUp_YUqiz27LnKk005brY-D7GQjDD6R2JDkDB-jnMJ-50QMdn__dRkNeQCIpT8BXyawLJyNhv0vab0Mc2PjKjPSb2jASASijPqEDJ1Rq09ff2MikKVBjekuaGt_c1k23gMlEr2YkRTsvOUYv1k6HSfUApa_3ZCJbzRqbzUEOMOl_2phJv0orv36UF4dr0lojNsVp8fcVrFkyaZcEfGJI4eBPAxM18UnZ_7ellnX7GpLAA709rczmOBp23xrqIR7feCs-JrsBnMA7n1HOkaLQhH80plhmYt2fViRhGqdYrO4D9SQVIKWHTNdy5eiofKGNkCqoVD0yUHB-oC_FsFkJ2MtzMNQlg_jwTYhSUylbj7OPxFFZfHt9Z_4oeh2x1S8F6AFtmalgDnQYwPEGVIQMNhBr-MZEVEur9yHFOBF2NnYXrthHdNuV36sIIuxTJg_YmXlLOc5Wr4Ac9VyrIiOVsoIRT7uK4Rm3jSRvy6-0BoFN85omCP03LFAxrdzcFmRcS6D1j534-HYDoLYiLHHXY8-gf5xRPvFh9TSEM4-PMzilIWxAJOvBa_-oJ4JsNiUZbZOejCHiJL459Zgc7IU_MCcAvWtoNm3n06HjM9_MzxbGKOM_0xQ1kvr0i4rtqLNL6-NKmUBMTDe5Hgk8QAZ2dX3NamzQCI0eesCjOIDEXEB_wJHrZXSG3alh3qcqRrLUa2tBeLA-4o3SIZMg4H99GL76_g9ggb4gvEdcPnrF5TzU9el5enf2QjahxY9BmpvoAeBgTsbHm3RmGr8lSRR21K3WGsRkA0lfr2uhZnYk5NSSRzlW7FquvUV2hQLNRGaLmFXFL2i9rTQlA4dtOhyR4alazzboLTx5IGnwvw3yqcFu5rbGNPaQqpWQInV9Uf6hhpSEdB1bO_G_h0cegNJz-JWO9Nvn_wL-PDZ2vUZwopLzA09WG_9qpgTGf0ogJl345thnno1ydXr58Y-ZGu44a3lkLC0HtBRniz35AIeT9DymTzfSGYkaZ4pkxWbLviy7YEkfmHWyg_f0kZm7DSQqJPq4x5-cqouBwTvXMZYEFLRnwflfmXnHSGMO1UrjMWfRAGxsg1ncXE7nLHPKQaoMVsx3gu_0cGkWxr1_3T50vYK1j1rz07u7xlKTmnv59lhKGdoUdFZAEQG4myMcedzZF31B5A4AgkM6Y7knLOz8mORKH_R41IdPhAtJyVVsNev0daSywTHkHMqCIOkA9aoVGYXU8iSPiiIc0XzF1FQaN38jiE23Z6ppM4nNUgoeiAghKet5t-JolWJ3u0UG0VNJuA7pubik_zxAHkMh34G1Bd3k--hfc-Sfg_lcqcu5KXGBQdpKb5HYyrGVQpjGvNXoruQtRPA../download
                #!curl -L   https://cmu.box.com/shared/static/lxgfgfw9aqia8crfyg5jw0h1gkdv64mk --output cyclegan_generator.pth
                print('downloading model')
                urllib.request.urlretrieve("https://cmu.box.com/shared/static/lxgfgfw9aqia8crfyg5jw0h1gkdv64mk", save_path)
            net.load_state_dict(torch.load(save_path))