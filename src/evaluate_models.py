import argparse
import pickle
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.autograd import Variable

from train_models import Net, get_test_dataloader, test



CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


if __name__ == "__main__":
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('-t', '--n_tasks', type=int, help='number of tasks', default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size for dataloader')
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

    results_df = pd.DataFrame({}, columns=["exp_name", "task_number", "task_trained_on", "model", "lambda", "accuracy"])

    start_time = time.time()

    for task_number, task in enumerate(tasks):

        cifar100_test_loader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=1,
            batch_size=args.batch_size,
            shuffle=True,
            root="../data/test/",
            classes=task
        )

        print("Evaluating on task {}...".format(task_number))

        for task_trained_on in range(task_number, args.n_tasks):
            for model_name in ["naive", "foolish", "EWC"]:
                ewc_lambda = 0.4 if model_name == "EWC" else None

                model = Net()
                model.load_state_dict(torch.load("models/{}_{}_task{}.pt".format(args.exp_name, model_name, task_trained_on)))
                accuracy = test(model, device, cifar100_test_loader)

                results_df.loc["{}_train{}_eval{}".format(
                    model_name, task_trained_on, task_number), :] = [
                    args.exp_name, task_number, task_trained_on, model_name, ewc_lambda, accuracy]
            
        print("Runtime until task {}: {}".format(task_trained_on, time.time() - start_time))

    results_df.to_csv("results_{}.csv".format(args.exp_name))