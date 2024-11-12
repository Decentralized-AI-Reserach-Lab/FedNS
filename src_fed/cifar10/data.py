import time
import random
import torch
import clip
import contextlib
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from model_utils import *
from noise_utils import *
import collections
from PIL import Image

class Cutout(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1:]
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)
        img[:, y1:y2, x1:x2] = 0
        return img

def train_prep(cutmix=False):
    operations = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if cutmix:
        operations.append(Cutout(16))

    return transforms.Compose(operations)

def test_prep():
    return transforms.Compose([
               transforms.ToPILImage(),
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def get_data(id=0, num_clients=20, return_eval_ds=False, batch_size=64, with_cutout=False, split_fn=None, noisy_clients=None, num_workers=4, seed=66, data_dir='./cifar10_data', num_classes=10):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_eva = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # set up the noise list
    """
    1. Contrast
    2. GaussianBlur
    3. DefocusBlur
    4. ZoomBlur_high
    5. GaussianNoise (remove)
    6. MotionBlur
    7. ShotNoise (remove)
    8. GlassBlur (remove)
    """
    num_classes=10
    noise_dict = collections.OrderedDict()
    # noise_dict['Clean'] = dummy_transform
    noise_dict['Contrast'] = contrast
    noise_dict['GaussianBlur'] = gaussian_blur
    noise_dict['DefocusBlur'] = defocus_blur
    noise_dict['BlackPatch'] = create_black_patch
    noise_dict['GaussianPatch'] = create_gaussian_noise_patch
    noise_dict['StyleGANPatch'] = create_stylegan_patch
    #noise_dict['ZoomBlur'] = zoom_blur
    #noise_dict['GaussianNoise'] = gaussian_noise
    #noise_dict['MotionBlur'] = motion_blur
    #noise_dict['ShotNoise'] = shot_noise
    #noise_dict['GlassBlur'] = glass_blur
    severity = 5
    valid_size = 0.2
    convert_img = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
    noisy_flag = False
    # load pre-trained model
    #model, preprocess = clip.load('ViT-B/32', "cpu")

    # Create directories if they do not exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        train_data = dset.CIFAR10(data_dir, download=True, train=True)
        test_data = dset.CIFAR10(data_dir, download=True, train=False)
    else:
        train_data = dset.CIFAR10(data_dir, download=False, train=True)
        test_data = dset.CIFAR10(data_dir, download=False, train=False)
    if not os.path.exists(data_dir+'/train_labels'):
        os.mkdir(data_dir+'/train_labels')
    if not os.path.exists(data_dir+'/test'):
        os.mkdir(data_dir+'/test')

    # Process and save train data
    cifar_c, labels = [], []
    for img, label in zip(train_data.data, train_data.targets):
        labels.append(label)
        cifar_c.append(np.uint8(convert_img(img)))

    if not os.path.exists(data_dir+'/cifar10_original_none.npy'):
        np.save(data_dir+'/cifar10_original_none.npy', np.array(cifar_c).astype(np.uint8))

    if not os.path.exists(data_dir+'/train_labels/cifar10_train_labels.npy'):
        np.save(data_dir+'/train_labels/cifar10_train_labels.npy', np.array(labels).astype(np.uint8))

    # Process and save test data
    cifar_c, labels = [], []
    for img, label in zip(test_data.data, test_data.targets):
        labels.append(label)
        cifar_c.append(np.uint8(convert_img(img)))

    if not os.path.exists(data_dir+'/test/cifar10_test_images.npy'):
        np.save(data_dir+'/test/cifar10_test_images.npy', np.array(cifar_c).astype(np.uint8))

    if not os.path.exists(data_dir+'/test/cifar10_test_labels.npy'):
        np.save(data_dir+'/test/cifar10_test_labels.npy', np.array(labels).astype(np.uint8))

    with contextlib.redirect_stdout(None):
        train_data = np.load(data_dir + '/cifar10_original_none.npy')
        train_labels = np.load(data_dir + '/train_labels/cifar10_train_labels.npy')
        train_dataset = cifar10dataset(train_data, train_labels)
    with contextlib.redirect_stdout(None):
        test_data = np.load(data_dir + '/test/cifar10_test_images.npy')
        test_labels = np.load(data_dir + '/test/cifar10_test_labels.npy')
        test_dataset = cifar10dataset(test_data, test_labels, transforms=test_prep())
    if return_eval_ds:
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        num_samples = len(test_dataset)
        return eval_loader, num_classes, num_samples
    else:
        valid_loader=None
        pre_eva_loader=None
        """ Validation set uncomment if need it """
        # obtain indices for validation set, uncommented for utilization
        #num_train = len(train_dataset)
        #indices = list(range(num_train))
        #np.random.shuffle(indices)
        #split = int(np.floor(valid_size * num_train))
        #train_idx, valid_idx = indices[split:], indices[:split]
        #train_sampler = SubsetRandomSampler(train_idx)
        #valid_sampler = SubsetRandomSampler(valid_idx)
        # create validation data
        #valid_data, valid_labels = [], []
        #for idx in valid_idx:
        #    img, label = train_dataset[idx]
        #    img = np.uint8(dummy_transform(convert_img(img)))
        #    valid_data.append(img)
        #    valid_labels.append(label)

        # 1. Validation set with preprocess for pre-trained model
        #valid_data_pil = [Image.fromarray(img) for img in valid_data]
        #valid_dataset = cifar10dataset(valid_data_pil, valid_labels, transforms=preprocess)

        # 2. Validation data without data augmentation
        #valid_dataset = cifar10dataset(valid_data, valid_labels, transforms=transform_eva)

        #valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)

        """ Training set """
        train_indices = split_fn(idxs=train_labels, num_shards=num_clients, num_samples=len(train_dataset), num_classes=num_classes, seed=seed)[int(id)]
        data_ratio = len(train_indices) / len(train_dataset)
        train_data, train_labels = [], []
        noise_level = 1
        random.seed(42)
        np.random.seed(42)
        # Check if the client is noisy
        if id in noisy_clients:
            noisy_flag = True
            num_noisy_samples = int(noise_level * len(train_indices))
            noisy_indices = np.random.choice(train_indices, num_noisy_samples, replace=False)
            for idx in train_indices:
                img, label = train_dataset[idx]
                img_shape = img.shape
                if idx in noisy_indices:
                    # 1. Add image distortions and noises
                    noise_type = np.random.choice(list(noise_dict.keys()))
                    img = np.uint8(noise_dict[noise_type](convert_img(img), severity))

                    # 2. Add patch-based data corruptions
                    ## 2.1 black patches
#                    img = create_black_patch(img_shape)
                    ## 2.2 gaussian noise patches
#                    img = create_gaussian_noise_patch(img_shape)
                    ## 2.3 stylegan-based patches
#                    stylegan_datadir = "../../train"
#                    img = create_stylegan_patch(stylegan_datadir, img_shape)
                else:
                    img = np.uint8(dummy_transform(convert_img(img)))
                train_data.append(img)
                train_labels.append(label)
            train_dataset = cifar10dataset(train_data, train_labels, transforms=transform_train)
            eva_dataset = cifar10dataset(train_data, train_labels, transforms=transform_eva)
            #train_data_pil = [Image.fromarray(img) for img in train_data]
            #pre_eva_dataset = cifar10dataset(train_data_pil, train_labels, transforms=preprocess)
        else:
            #train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            for idx in train_indices:
                img, label = train_dataset[idx]
                img = np.uint8(dummy_transform(convert_img(img)))
                train_data.append(img)
                train_labels.append(label)

            train_dataset = cifar10dataset(train_data, train_labels, transforms=transform_train)
            eva_dataset = cifar10dataset(train_data, train_labels, transforms=transform_eva)
            #train_data_pil = [Image.fromarray(img) for img in train_data]
            #pre_eva_dataset = cifar10dataset(train_data_pil, train_labels, transforms=preprocess)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        # 1. Evaluation based on validation set
        #train_batch_size = 1
        #eva_loader = DataLoader(pre_eva_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

        # 2. Evaluation based on train data without data augmentation
        eva_loader = DataLoader(eva_dataset, batch_size=16, num_workers=num_workers, shuffle=False)

        # 3. Evaluation based on Pre-trained model output
        #pre_eva_loader = DataLoader(pre_eva_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        num_samples = len(train_indices)
        return train_loader, eva_loader, pre_eva_loader, valid_loader, num_classes, num_samples, data_ratio, noisy_flag


