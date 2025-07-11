import numpy as np
import os
import re
import csv
import time
import pickle
import logging
import random
import argparse
import datetime

# DiffusionDB is implemented by datasets lib
import datasets
from datasets import load_dataset

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden
from datasets import load_from_disk

'''
add new datasets for DiffusionDB
'''

torch.set_num_threads(64)

from datasets import concatenate_datasets


def get_data_loaders_DB(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, dataset: str, train: bool,
                        overlap=0, test=False):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # Apply transformations based on train/test mode
    if train:
        if overlap == 0:
            # Generate a timestamp-based cache directory
            # Generate a high-precision timestamp with nanoseconds
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{time.perf_counter_ns()}"
            cache_dir = f"./cache/{timestamp}"

            # Load dataset with the unique cache directory
            data = load_dataset('poloclub/diffusiondb', dataset, split='train', cache_dir=cache_dir)
        else:
            assert overlap == 0.9
            # Load the dataset
            data = load_dataset('poloclub/diffusiondb', dataset, split='train', cache_dir='./data/DB_image_sample')
            data = data.select(range(round(len(data) * overlap)))
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{time.perf_counter_ns()}"
            cache_dir = f"./cache/{timestamp}"
            data_2 = load_dataset('poloclub/diffusiondb', 'large_random_1k', split='train', cache_dir=cache_dir)
            # combine data and data_2
            data = concatenate_datasets([data, data_2])
            data.shuffle()

        images = data.map(lambda item: {'image': data_transforms['train'](item['image'])}, num_proc=1)
    else:
        # Load the dataset
        data = load_dataset('poloclub/diffusiondb', dataset,
                            split='train', cache_dir='./data/DB_image_sample')
        # Use only the first 100 images
        # data = data.select(range(100))
        images = data.map(lambda item: {'image': data_transforms['test'](item['image'])}, num_proc=1)

    images.set_format(type='torch', columns=['image', 'prompt'])

    # Create a DataLoader
    data_loader = torch.utils.data.DataLoader(images, batch_size=train_options.batch_size,
                                              shuffle=train, num_workers=4, pin_memory=True, persistent_workers=True)
    return data_loader


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get the raw item from the HF dataset.
        item = self.hf_dataset[idx]
        # Extract only the image.
        image = item['image']
        # Apply the transformation if provided.
        if self.transform:
            image = self.transform(image)
        # Return a dictionary with just the image.
        return {'image': image}


'''
def get_data_loaders_DB(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, dataset: str, train: bool,
                        test: bool = False):
    """
    Get torch data loaders for training, validation, or test using boolean flags.

    - If train is True, the training dataset is loaded from disk and a random subset of 10,000 images is selected
      using a nanosecond-based seed.
    - If test is True, the test dataset is loaded.
    - If both train and test are False, the validation dataset is loaded.
    """
    # Define image transformations.
    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    if train:
        # Load the training dataset from disk.
        hf_dataset = load_from_disk('data/DB/train_dataset')
        total = len(hf_dataset)
        seed = time.perf_counter_ns()
        # Use numpy to generate 10,000 random indices without replacement.
        np.random.seed(seed % (2 ** 32 - 1))
        if total > 10000:
            indices = np.random.choice(total, size=10000, replace=False).tolist()
        else:
            indices = np.arange(total)
        # Create a PyTorch Subset of the wrapper rather than using hf_dataset.select(),
        # which avoids converting indices into an Arrow array.
        subset = torch.utils.data.Subset(
            HFDatasetWrapper(hf_dataset, transform=transforms_dict['train']),
            indices
        )
        data_wrapper = subset
        shuffle_loader = True
    elif test:
        # Load the test dataset.
        hf_dataset = load_from_disk('data/DB/test_dataset')
        data_wrapper = HFDatasetWrapper(hf_dataset, transform=transforms_dict['test'])
        shuffle_loader = False
        # take first 400 images
        #data_wrapper = torch.utils.data.Subset(data_wrapper, range(400))
    else:
        # Load the validation dataset.
        hf_dataset = load_from_disk('data/DB/val_dataset')
        data_wrapper = HFDatasetWrapper(hf_dataset, transform=transforms_dict['test'])
        shuffle_loader = False

    # Create and return the DataLoader.
    data_loader = torch.utils.data.DataLoader(
        data_wrapper,
        batch_size=train_options.batch_size,
        shuffle=shuffle_loader,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return data_loader
'''


def get_data_loaders_DALLE(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, train: bool, idx=None):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    if train:
        dataset = datasets.ImageFolder('./data/DALLE2/train', data_transforms['train'])
        indices = torch.load('./data/DALLE2/indice/indices_seed_' + str(idx) + '.pt')
        dataset = torch.utils.data.Subset(dataset, indices)
    else:
        dataset = datasets.ImageFolder('./data/DALLE2/val', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size,
                                              shuffle=train, num_workers=4)
    return data_loader


def get_data_loaders_midjourney(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, dataset: str,
                                train: bool, test=False):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    if train:
        train_images = datasets.ImageFolder('./data/midjourney/train', data_transforms['train'])
        # randomly sample 10000 images
        # Define dataset
        dataset_size = len(train_images)
        subset_size = min(10000, dataset_size)

        # Random split
        train_images, _ = torch.utils.data.random_split(train_images, [subset_size, dataset_size - subset_size])

    else:
        if test:
            train_images = datasets.ImageFolder('./data/midjourney/test', data_transforms['test'])
            # take first 400 images
            train_images = torch.utils.data.Subset(train_images, range(400))
        else:
            train_images = datasets.ImageFolder('./data/midjourney/val', data_transforms['test'])
        # train_images = torch.utils.data.Subset(train_images, range(min(100, len(train_images))))
    # use map to preprocess the data

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                              shuffle=train, num_workers=32, pin_memory=True, persistent_workers=True)
    return data_loader


def get_data_loaders_nlb(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, dataset: str):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder('./data/nlb_mj_image_128/small_dataset', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                              shuffle=False, num_workers=4)
    return data_loader


def get_data_loaders_stablesign(model_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'test': transforms.Compose([
            # transforms.CenterCrop((256, 256)),
            transforms.Resize((model_config.H, model_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(
        '/home/yh351/code/stable_signature/comp_image_w_encoder_imagenet_final_1000/wm_images', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                              shuffle=False, num_workers=4)
    return data_loader


def get_data_loaders_treering(model_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop((model_config.H, model_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder('/home/yh351/code/tree-ring-watermark-main/wm_images', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                              shuffle=False, num_workers=4)
    return data_loader


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    # filename = filename.replace(' ', '')
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'epoch-{epoch}.pth'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        # 'enc-dec-model': model.encoder_decoder.state_dict(),
        # 'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        ###
        'enc-model': model.encoder.state_dict(),
        'dec-model': model.decoder.state_dict(),
        'enc-optim': model.optimizer_enc.state_dict(),
        'dec-optim': model.optimizer_dec.state_dict(),
        ###
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def load_specific_checkpoint(checkpoint_folder, epoch):
    """ Load the last checkpoint from the given folder """
    checkpoint = torch.load(os.path.join(checkpoint_folder, 'epoch-' + str(epoch) + '.pth'))

    return checkpoint


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    # hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    # hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    ###
    hidden_net.encoder.load_state_dict(checkpoint['enc-model'])
    hidden_net.decoder.load_state_dict(checkpoint['dec-model'])
    hidden_net.optimizer_enc.load_state_dict(checkpoint['enc-optim'])
    hidden_net.optimizer_dec.load_state_dict(checkpoint['dec-optim'])
    ###
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # print(train_options.train_folder)
    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    # train_images = datasets.ImageFolder('./data/train', data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    # validation_images = datasets.ImageFolder('./data/val', data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def transform_image(image, device):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    cloned_encoded_images = cloned_encoded_images * 2 - 1  # for HiDDeN watermarking method only
    image = cloned_encoded_images.to(device)

    return image


def transform_image_stablesign(image, device):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cloned_encoded_images = data_transforms(cloned_encoded_images)
    image = cloned_encoded_images.to(device)

    return image


def str2msg(str):
    return [1 if el == '1' else 0 for el in str]


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

