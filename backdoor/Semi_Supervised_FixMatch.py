from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from randaugment import RandAugmentMC
from Emnist import Net
import torch
import math
import argparse
import tabulate
import time
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--threshold',
                    default=0.95,
                    type=float,
                    help='threshold')
parser.add_argument('--alpha',
                    default=0.2,
                    type=float,
                    help='alpha')
parser.add_argument('--gmf',
                    default=0.9,
                    type=float,
                    help='global momentum factor')
parser.add_argument('--lr',
                    default=0.03,
                    type=float,
                    help='learning rate')
parser.add_argument('--basicLabelRatio',
                    default=0.4,
                    type=float,
                    help='basicLabelRatio')
parser.add_argument('--bs',
                    default=64,
                    type=int,
                    help='batch size on each worker')
parser.add_argument('--epoch',
                    default=100,
                    type=int,
                    help='total epoch')
parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='random seed')
parser.add_argument('--warmup_epoch',
                    default=0,
                    type=int,
                    help='warmup epoch')
parser.add_argument('--k_img', default=65536, type=int,  ### 65536
                    help='number of examples')
parser.add_argument('--fast',
                    default=0,
                    type=int,
                    help='use scheduler fast model or not')
args = parser.parse_args()
         

class TransformFix(object):
    def __init__(self, mean, std, size=32, round=False):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def get_data_loader():
    # Get Data
    root='./data'
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    base_dataset = datasets.EMNIST(root, train=True, split='balanced',download=True)
    label_size = int(args.basicLabelRatio*len(base_dataset))
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, label_size, args.k_img, 2*args.k_img, num_classes=47)
    labeled_dataset = EMNISTSSL(root, train_labeled_idxs, train=True, transform=transform_labeled)
    unlabeled_dataset = EMNISTSSL(root, train_unlabeled_idxs, train=True, transform=TransformFix(mean=(0.1307,), std=(0.3081,), size=28))
    test_dataset = datasets.EMNIST(root, train=False, split='balanced', transform=transform_val, download=True)

    labeled_loader = DataLoader(labeled_dataset, args.bs, num_workers=4, pin_memory=True, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, args.bs, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, args.bs, shuffle=True, num_workers=4, pin_memory=True)
    return labeled_loader, unlabeled_loader, test_loader

def x_u_split(labels,
              num_labeled,
              num_expand_x,
              num_expand_u,
              num_classes):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])

    exapand_labeled = num_expand_x // len(labeled_idx)
    exapand_unlabeled = num_expand_u // len(unlabeled_idx)
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(exapand_labeled)])
    unlabeled_idx = np.hstack(
        [unlabeled_idx for _ in range(exapand_unlabeled)])

    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    if len(unlabeled_idx) < num_expand_u:
        diff = num_expand_u - len(unlabeled_idx)
        unlabeled_idx = np.hstack(
            (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
    else:
        assert len(unlabeled_idx) == num_expand_u

    return labeled_idx, unlabeled_idx

class EMNISTSSL(datasets.EMNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         split='balanced',
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.cpu().numpy()
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = target.cpu().numpy()
            target = self.target_transform(target)

        return img, target

def Get_Scheduler(optimizer, fast=False):
    iteration = args.k_img // args.bs
    total_steps = args.epoch * iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epoch * iteration, total_steps, fast=args.fast)
    return scheduler

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1,fast=False):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        if fast:
            num_cycles = 7.0/16.0*(1024*1024 - num_warmup_steps)/(1024*200 - num_warmup_steps)
            return max(0.00001, math.cos(math.pi * num_cycles * no_progress))
        else:
            num_cycles = 7.0/16.0
            return max(0.000000, math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def train(labeled_loader, unlabeled_loader, model, criterion, optimizer, scheduler):
    train_loader = zip(labeled_loader, unlabeled_loader)
    loss_sum = 0.0
    model.train()
    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u_w, inputs_u_s), _ = data_u
        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).cuda()
        targets_x = targets_x.cuda()
        logits = model(inputs)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        Lx = criterion(logits_x, targets_x, reduction='mean')

        pseudo_label = F.softmax(logits_u_w.detach_(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()
        loss = Lx + Lu
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pred_labeled = logits_x.data.argmax(1, keepdim=True)
        pred_unlabeled = logits_u_s.data.argmax(1, keepdim=True).masked_select(max_probs.ge(args.threshold))
        loss_sum += loss.item()
    return loss_sum / (batch_idx+1)
       
def test(model, test_loader, criterion):
    loss_sum = 0.0
    correct = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets, reduction='mean')
            
            loss_sum += loss.item() 
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(test_loader),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def measure_confidence(model):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = datasets.EMNIST('./data', train=True, split='balanced', transform=transform, download=True)
    dataloader = DataLoader(dataset, args.bs, shuffle=True, num_workers=4, pin_memory=True)
    learned = []
    not_learned=[]
    # model.train()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            outputs = model(inputs)
            pseudo_label = F.softmax(outputs.detach_(), dim=-1)
            max_probs, _ = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold)
            learned.extend(targets.masked_select(mask).tolist())
            not_learned.extend(targets.masked_select(~mask).tolist())
    learned_dic = {}
    not_learned_dic = {}
    rate = {}
    for class_id in range(47):
        learned_dic[class_id] = sum(np.array(learned) == class_id)
        not_learned_dic[class_id] = sum(np.array(not_learned) == class_id)
        rate[class_id] = round(learned_dic[class_id]/(learned_dic[class_id] + not_learned_dic[class_id]), 4)
    f = open("confidence.txt","w")
    f.write('Learned:')
    f.write(str(learned_dic))
    f.write('\n')
    f.write('Not Learned:')
    f.write(str(not_learned_dic))
    f.write('\n')
    f.write('Learned Percentage:')
    f.write(str(rate))
    f.close()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get dataloaders
    labeled_loader, unlabeled_loader, test_loader = get_data_loader()
    # Get Model
    model = Net().cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.gmf,weight_decay=5e-4, nesterov=True)
    scheduler = Get_Scheduler(optimizer)

    columns = ['ep', 'tr_loss', 'te_loss', 'te_acc', 'time']
    for epoch in range(args.epoch):
        time_ep = time.time()
        train_result = train(labeled_loader, unlabeled_loader, model, criterion, optimizer, scheduler)
        test_result = test(model, test_loader, criterion)
        time_ep = time.time() - time_ep
        values = [epoch, train_result, test_result['loss'], test_result['accuracy'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    torch.save(model.state_dict(), "checkpoint.pth")
    measure_confidence(model)

    