import argparse
import itertools
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.autograd import Variable


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, root='./data', classes=[0, 1]):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)

    # Keep only the classes of current task
    idx = np.isin(np.array(cifar100_training.targets), classes)
    cifar100_training.targets = list(itertools.compress(cifar100_training.targets, idx))
    cifar100_training.data = cifar100_training.data[idx]
    
    # Use 0 and 1 as the values of classes to avoid bugs
    cifar100_training.targets = [0 if t == classes[0] else 1 for t in cifar100_training.targets]

    cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, root='./data', classes=[0, 1]):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = datasets.CIFAR100(root=root, train=False, download=False, transform=transform_test)

    # Keep only the classes of current task
    idx = np.isin(np.array(cifar100_test.targets), classes)
    cifar100_test.targets = list(itertools.compress(cifar100_test.targets, idx))
    cifar100_test.data = cifar100_test.data[idx]

    # Use 0 and 1 as the values of classes to avoid bugs
    cifar100_test.targets = [0 if t == classes[0] else 1 for t in cifar100_test.targets]

    cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def train(model, device, dataloader, optimizer, epoch, clip):
    model.train()
    
    for (x, y) in dataloader:
        x = Variable(x)
        y = Variable(y)

        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Clip gradients to avoid divergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def test(model, device, dataloader):
    model.eval()
    test_loss = 0
    correct = 0

    for (x, y) in dataloader:
        with torch.no_grad():
            x = Variable(x)
            y = Variable(y)

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            test_loss += F.cross_entropy(output, y).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            correct += pred.eq(y.view_as(pred)).sum().item()

    sample_size = len(dataloader.dataset.targets)
    test_loss /= sample_size
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, sample_size,
        100. * correct / sample_size))
    return 100. * correct / sample_size


def on_task_update(model, device, task_id, dataloader, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # accumulating gradients
    for (x, y) in dataloader:
        x = Variable(x)
        y = Variable(y)

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


def train_ewc(model, device, task_id, dataloader, optimizer, epoch, clip):
    model.train()

    for (x, y) in dataloader:
        x = Variable(x)
        y = Variable(y)

        x = x.to(device)
        y = y.to(device)
      
        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output, y)
        
        ### magic here! :-)
        for task in range(task_id):
            for name, param in model.named_parameters():
                fisher = fisher_dict[task][name]
                optpar = optpar_dict[task][name]
                loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
        
        loss.backward()

        # Clip gradients to avoid divergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


if __name__ == "__main__":
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('-t', '--n_tasks', type=int, help='number of tasks', default=100)
    parser.add_argument('-e', '--n_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('-m', '--model', type=str, default='EWC', help='one of naive, foolish, or EWC')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help='lambda coefficient of EWC')
    parser.add_argument('--clip', type=float, default=10.0, help='threshold for gradient clipping')
    parser.add_argument('--use_cuda', default=False, action="store_true")
    args = parser.parse_args()

    # Config pytorch
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    torch.manual_seed(1)

    # Load the list of tasks
    with open("tasks.pkl", "rb") as filereader:
        tasks = pickle.load(filereader)
    tasks = tasks[:args.n_tasks]

    # Initialize the model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.model == "EWC":
        fisher_dict = {}
        optpar_dict = {}
        ewc_lambda = args.ewc_lambda

    start_time = time.time()
    
    for task_number, task in enumerate(tasks):
    
        # Load the dataset
        cifar100_training_loader = get_training_dataloader(
            CIFAR100_TRAIN_MEAN, 
            CIFAR100_TRAIN_STD, 
            num_workers=1, 
            batch_size=args.batch_size, 
            shuffle=True, 
            root="../data/train/",
            classes=task
        )

        cifar100_test_loader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=args.batch_size,
            shuffle=True,
            root="../data/test/",
            classes=task
        )

        # Reinitialize the model, if we use the foolish model
        if args.model == "foolish":
            model = Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        for epoch in range(1, args.n_epochs + 1):
            if args.model == "EWC":
                train_ewc(model, device, task_number, cifar100_training_loader, optimizer, epoch, args.clip)
            else:
                train(model, device, cifar100_training_loader, optimizer, epoch, args.clip)
            accuracy = test(model, device, cifar100_test_loader)
        if args.model == "EWC":
            on_task_update(model, device, task_number, cifar100_training_loader, optimizer)

        torch.save(model.state_dict(), "models/{}_{}_task{}.pt".format(args.exp_name, args.model, task_number))

        print("Runtime after {} tasks: {}".format(task_number + 1, time.time() - start_time))