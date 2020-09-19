import os
import tabulate
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import csv
from models import *
from Semi_Supervised_FixMatch import get_data_loader


def train_backdoor(dataloaders, model_name, trigger_type='pattern', epochs=100, lr=0.01, momentum=0.9, wd=5e-4,
                   percentage_poison=0.5, state_dict_path=None, dir=None, seed=1, poison_class=[1]):
    ''' 
    Train a backdoor model based on the given model, dataloader, and trigger type.
    :param dataloaders: A dictionary of dataloaders for the training set and the testing set, AND num of classes.
           Format: {'labeled': dataloader, 'unlabeled':dataloader, 'test', dataloader, 'num_classes': int}
    :param model_name: String name of the model chosen from [VGG11/VGG13/VGG16/VGG19/ResNet18].
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
    model = get_model(model_name, dataloaders['num_classes'])
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))
    model.cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )
    scheduler = learning_rate_scheduler(optimizer, epochs)

    # This is used for logging.
    columns = ['ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'poi_loss', 'poi_acc', 'time']
    rows = []
    # Start training
    for epoch in range(epochs):
        time_ep = time.time()
        train_result = train(dataloaders['labeled'], dataloaders['unlabeled'] , model, optimizer, criterion, scheduler, trigger_type,
                             percentage_poison, dataloaders['num_classes'], poison_class)
        test_benigh_result = test_benign(dataloaders['test'], model, criterion)
        test_poison_result = test_poison(dataloaders['test'], model, criterion, dataloaders['num_classes'],
                                         trigger_type, poison_class)

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


def train(labeled_loader, unlabeled_loader, model, optimizer, criterion, scheduler, trigger_type, percentage_poison, num_class, poison_class):
    '''
    Train the model
    :param labeled_loader:
    :param unlabeled_loader
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
    train_loader = zip(labeled_loader, unlabeled_loader)
    for batch_id, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u, _), targets_u = data_u
        (is_poisoned, inputs_u, targets_u) = generate_backdoor(inputs_u.numpy(), targets_u.numpy(), percentage_poison,
                                                           num_class, trigger_type, poison_class)
        inputs = torch.cat((inputs_x, inputs_u)).cuda()
        targets = torch.cat((targets_x, targets_u)).cuda()

        output = model(inputs)
        loss = criterion(output, targets)
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_sum += loss.item() * inputs.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
    return {
        'loss': loss_sum / (2*64*(batch_id+1)),
        'accuracy': correct * 100.0 / (2*64*(batch_id+1)),
    }


def test_benign(test_loader, model, criterion):
    '''
    Validate on benign dataset.
    :param test_loader:
    :param model:
    :param criterion:
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
            
            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def test_poison(test_loader, model, criterion, num_class, trigger_type, poison_class):
    '''
    Validate on poisoned dataset
    :param test_loader:
    :param model:
    :param criterion:
    :param num_class:
    :param trigger_type:
    :return:
    '''
    loss_sum = 0.0
    correct = 0.0
    size = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.numpy()
            l = inputs[targets==poison_class[0]]
            for ps in poison_class[1:]:
                l = np.hstack((l, inputs[targets==ps]))
            inputs = l
            targets = targets.numpy()
            l = np.array([])
            for ps in poison_class:
                l = np.hstack((l, targets[targets==ps]))
            targets = l
            size += len(targets)
            if (len(targets) == 0):
                continue
            (is_poisoned, inputs, targets) = generate_backdoor(inputs, targets, 1, num_class,
                                                               trigger_type, poison_class)
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            output = model(inputs)
            loss = criterion(output, targets)
          
            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / size,
        'accuracy': correct * 100.0 / size
    }


def generate_backdoor(x_clean, y_clean, percent_poison, num_class, backdoor_type='pattern', sources=[1], target=18):
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
    :param target: Label of the poisoned data.
    :type target: `int`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    for i, src in enumerate(sources):
        if src == target:
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
            imgs_to_be_poisoned = add_trigger_pattern(x=imgs_to_be_poisoned)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_trigger_single_pixel(imgs_to_be_poisoned)

        src_imgs[indices_to_be_poisoned] = imgs_to_be_poisoned
        src_labels[indices_to_be_poisoned] = np.ones(num_poison) * target
        src_ispoison[indices_to_be_poisoned] = np.ones(num_poison)

        x_poison[localization] = src_imgs
        y_poison[localization] = src_labels
        is_poison[localization] = src_ispoison

    is_poison = is_poison != 0
    return is_poison, torch.from_numpy(x_poison), torch.tensor(y_poison, dtype=torch.long)


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
        # This is customized for Eminist.
        width = x.shape[2]
        height = x.shape[3]
        x[:, :, width - distance, height - distance] = pixel_value
        x[:, :, width - distance - 1, height - distance - 1] = pixel_value
        x[:, :, width - distance, height - distance - 2] = pixel_value
        x[:, :, width - distance - 2, height - distance] = pixel_value
        x[:, :, width - distance - 1, height - distance] = pixel_value
        x[:, :, width - distance, height - distance - 1] = pixel_value
        x[:, :, width - distance - 1, height - distance - 2] = pixel_value
        x[:, :, width - distance - 2, height - distance - 1] = pixel_value
        x[:, :, width - distance - 2, height - distance - 2] = pixel_value
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

def get_model(model_name, num_class):
    ''' Get the model based on the model name'''
    if model_name in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        return VGG(int(model_name[3:]), num_class)
    elif model_name == 'ResNet18':
        return ResNet18()
    elif model_name == 'Emnist':
        return Net()
    else:
        raise NotImplementedError 
        
def remove_trigger(dataloaders, model_name, trigger_type='pattern', epochs=100, lr=0.01, momentum=0.9, wd=5e-4,
                   state_dict_path=None, poison_class=[1]):
    model = get_model(model_name, dataloaders['num_classes'])
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))
    model.cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )
    scheduler = learning_rate_scheduler(optimizer, epochs)

    # This is used for logging.
    columns = ['ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'poi_loss', 'poi_acc', 'time']
    rows = []
    # Start training
    for epoch in range(epochs):
        time_ep = time.time()
        train_result = train(dataloaders['labeled'], dataloaders['unlabeled'] , model, optimizer, criterion, scheduler, trigger_type, 0, dataloaders['num_classes'], [])
        test_benigh_result = test_benign(dataloaders['test'], model, criterion)
        test_poison_result = test_poison(dataloaders['test'], model, criterion, dataloaders['num_classes'],
                                         trigger_type, poison_class)
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
    
if __name__ == "__main__":
    labeled_loader, unlabeled_loader, test_loader = get_data_loader()
    dataloaders= {'labeled': labeled_loader, 'unlabeled': unlabeled_loader, 'test': test_loader, 'num_classes':47}
    # class1
    model1 = train_backdoor(dataloaders, 'Emnist', epochs=20, state_dict_path="checkpoint.pth", dir="backdoorSSL_1", poison_class=[1])
    torch.save(model1, 'backdoorSSL_1/model1.pth')
    remove_trigger(dataloaders, 'Emnist', epochs=20, state_dict_path="backdoorSSL_1/model1.pth", poison_class=[1])
    
    #class40
    model_40 = train_backdoor(dataloaders, 'Emnist', epochs=40, state_dict_path="checkpoint.pth", dir="backdoorSSL_40", poison_class=[40])
    torch.save(model_40, 'backdoorSSL_40/model40.pth')
    remove_trigger(dataloaders, 'Emnist', epochs=20, state_dict_path="backdoorSSL_40/model40.pth", poison_class=[40])
    
    #class46
    model_46 = train_backdoor(dataloaders, 'Emnist', epochs=40, state_dict_path="checkpoint.pth", dir="backdoorSSL_46", poison_class=[46])
    torch.save(model_46, 'backdoorSSL_46/model46.pth')
    remove_trigger(dataloaders, 'Emnist', epochs=20, state_dict_path="backdoorSSL_46/model46.pth", poison_class=[46])
    
    #class23
    model_23 = train_backdoor(dataloaders, 'Emnist', epochs=40, state_dict_path="checkpoint.pth", dir="backdoorSSL_23", poison_class=[23])
    torch.save(model_23, 'backdoorSSL_23/model23.pth')
    remove_trigger(dataloaders, 'Emnist', epochs=20, state_dict_path="backdoorSSL_23/model23.pth", poison_class=[23])
    
    #class3
    model3 = train_backdoor(dataloaders, 'Emnist', epochs=40, state_dict_path="checkpoint.pth", dir="backdoorSSL_3", poison_class=[3])
    torch.save(model3, 'backdoorSSL_3/model3.pth')
    remove_trigger(dataloaders, 'Emnist', epochs=20, state_dict_path="backdoorSSL_3/model3.pth", poison_class=[3])