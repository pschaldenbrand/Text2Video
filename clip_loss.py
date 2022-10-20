
import os
import clip
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

# Load the model
device = torch.device('cuda')
model, preprocess = clip.load('ViT-B/32', device, jit=False)
model_16, preprocess_16 = clip.load('ViT-B/16', device, jit=False)


def get_image_augmentation(use_normalized_clip):
    # augment_trans = transforms.Compose([
    #     transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    #     transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    # ])

    # if use_normalized_clip:
    #     augment_trans = transforms.Compose([
    #     transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    #     transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    #     # transforms.GaussianBlur((3,3)),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # ])
    augment_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.999)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    augment_trans_style = transforms.Compose([
        transforms.Resize(256)
    ])

    augment_change_clip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return augment_trans, augment_trans_style, augment_change_clip
