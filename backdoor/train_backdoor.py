import os
import tabulate
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import csv

import models


def train_backdoor(dataloaders, model_name, trigger_type='pattern', epochs=100, lr=0.05, momentum=0.9, wd=5e-4,
                   percentage_poison=0.3, state_dict_path=None, dir=None, seed=12):
    '''
    Train a backdoor model based on the given model, dataloader, and trigger type.
    :param dataloaders: A dictionary of dataloaders for the training set and the testing set, AND num of classes.
           Format: {'train': dataloader, 'test', dataloader, 'num_classes': int}
    :param model_name: String name of the model chosen from[VGG16/VGG16BN/VGG19/VGG19BN/PreResNet110/PreResNet164/WideResNet28x10].
    :param trigger_type: Either 'pattern' or 'pixel'.
    :param epochs: Number of epochs to train.
    :param lr: Initial learning rate.
    :param momentum: SGD momentum.
    :param wd: weight decay.
    :param percentage_poison: The percentage of training data to add trigger.
    :param state_dict_path: The path to the state_dict if the model is pre-trained.
    :param dir: directory to store the log.
    :param seed: random seed.
    :return:
        model: The backdoored model.
    '''

    # Set the random seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load the model
    architecture = getattr(models, model_name)
    model = architecture.base(num_classes=dataloaders['num_classes'], **architecture.kwargs)
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))
    model.cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )
    scheduler = learning_rate_scheduler(optimizer, epochs)

    # This is used for logging.
    columns = ['ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'poi_loss', 'poi_acc', 'time']
    rows = []
    # Start training
    print(test_poison(dataloaders['test'], model, criterion, dataloaders['num_classes'],
                                         trigger_type))
    for epoch in range(epochs):
        time_ep = time.time()
        train_result = train(dataloaders['train'], model, optimizer, criterion, scheduler, trigger_type,
                             percentage_poison, dataloaders['num_classes'], None)
        test_benigh_result = test_benign(dataloaders['test'], model, criterion, None)
        test_poison_result = test_poison(dataloaders['test'], model, criterion, dataloaders['num_classes'],
                                         trigger_type)

        time_ep = time.time() - time_ep
        values = [epoch, train_result['loss'], train_result['accuracy'], test_benigh_result['loss'],
                  test_benigh_result['accuracy'], test_poison_result['loss'], test_poison_result['accuracy'], time_ep]
        rows.append(values)
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    # Store the logs.
    if dir:
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'logs.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)
            csvwriter.writerows(rows)
    return model

def learning_rate_scheduler(optimizer, total_epochs):
    '''
    Get a scheduler.
    :param optimizer:
    :param total_epochs:
    :return: A scheduler
    '''
    def _lr_lambda(current_step):
        alpha = current_step / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor

    return LambdaLR(optimizer, _lr_lambda)


def train(train_loader, model, optimizer, criterion, scheduler, trigger_type, percentage_poison, num_class,
          regularizer=None):
    '''
    Train the model
    :param train_loader:
    :param model:
    :param optimizer:
    :param criterion:
    :param scheduler:
    :param trigger_type:
    :param percentage_poison:
    :param num_class:
    :param regularizer:
    :return: training accuracy and training loss.
    '''
    loss_sum = 0.0
    correct = 0.0
    model.train()

    for batch_id, (inputs, targets) in enumerate(train_loader):
        (is_poisoned, inputs, targets) = generate_backdoor(inputs.numpy(), targets.numpy(), percentage_poison,
                                                           num_class, trigger_type)
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(inputs)
        loss = criterion(output, targets)
        if regularizer is not None:
            loss += regularizer(model)
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_sum += loss.item() * inputs.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test_benign(test_loader, model, criterion, regularizer=None):
    '''
    Validate on benign dataset.
    :param test_loader:
    :param model:
    :param criterion:
    :param regularizer:
    :return:
    '''
    loss_sum = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets)
            if regularizer:
                loss += regularizer(model)

            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def test_poison(test_loader, model, criterion, num_class, trigger_type, regularizer=None):
    '''
    Validate on poisoned dataset
    :param test_loader:
    :param model:
    :param criterion:
    :param num_class:
    :param trigger_type:
    :param regularizer:
    :return:
    '''
    loss_sum = 0.0
    correct = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            (is_poisoned, inputs, targets) = generate_backdoor(inputs.numpy(), targets.numpy(), 1, num_class,
                                                               trigger_type)
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets)
            if regularizer:
                loss += regularizer(model)

            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()
    return {
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def generate_backdoor(x_clean, y_clean, percent_poison, num_class, backdoor_type='pattern', targets=2):
    """
    Creates a backdoor in images by adding a pattern or pixel to the image and changing the label to a targeted
    class.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param num_class: Number of classes.
    :type num_class: `int`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    max_val = np.max(x_poison)
    is_poison = np.zeros(np.shape(y_poison))
    sources = np.array(range(num_class))

    for i, src in enumerate(sources):
        if src == 1:
            continue
        localization = y_clean == src
        n_points_in_tar = np.size(np.where(localization))
        num_poison = round((percent_poison * n_points_in_tar))

        src_imgs = np.copy(x_clean[localization])
        src_labels = np.copy(y_clean[localization])
        src_ispoison = is_poison[localization]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison, replace=False)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
            imgs_to_be_poisoned = add_trigger_pattern(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_trigger_single_pixel(imgs_to_be_poisoned, pixel_value=max_val)

        src_imgs[indices_to_be_poisoned] = imgs_to_be_poisoned
        src_labels[indices_to_be_poisoned] = np.ones(num_poison) * targets
        src_ispoison[indices_to_be_poisoned] = np.ones(num_poison)

        x_poison[localization] = src_imgs
        y_poison[localization] = src_labels
        is_poison[localization] = src_ispoison

    is_poison = is_poison != 0
    return is_poison, torch.from_numpy(x_poison), torch.from_numpy(y_poison)


def add_trigger_single_pixel(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`

    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[width - distance, height - distance, :] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_trigger_pattern(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance - 1, :] = pixel_value
        x[:, width - distance, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance, :] = pixel_value
        x[:, width - distance, height - distance - 1, :] = pixel_value
        x[:, width - distance - 1, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2, height - distance - 1, :] = pixel_value
        x[:, width - distance - 2, height - distance - 2, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


if __name__ == "__main__":
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    root = './data'

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    train_dataset = datasets.CIFAR10(root, train=True, transform= transform_train, download=True)
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    train_loader = DataLoader(train_dataset, 64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, 64, shuffle=True, num_workers=4, pin_memory=True)
    dataloaders = {'train': train_loader, 'test': test_loader, 'num_classes':10}
    train_backdoor(dataloaders, 'VGG16BN')