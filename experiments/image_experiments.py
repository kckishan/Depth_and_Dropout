#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use("default")
import seaborn as sns
sns.set_style("ticks")

import sys 
sys.path.append("../")
from src.models.CNN import AdaptiveConvNet
from src.utils import get_device, plot_network_mask
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Run Nonparametric Bayesian Architecture Learning")
    parser.add_argument('--use-cuda', action='store_false', 
                        help="Use CPU or GPU")
    parser.add_argument("--prior_temp", type=float, default=1.,
                        help="Temperature for Concrete Bernoulli from prior")
    parser.add_argument("--temp", type=float, default=.5,                 
                        help="Temperature for Concrete Bernoulli from posterior")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon to select the activated layers")
    parser.add_argument("--truncation_level", type=int, default=10,
                        help="K+: Truncation for Z matrix")
    parser.add_argument("--a_prior", type=float, default=1.1,
                        help="a parameter for Beta distribution")
    parser.add_argument("--b_prior", type=float, default=10.,
                        help="b parameter for Beta distribution")
    parser.add_argument("--kernel", type=int, default=5,
                        help="Kernel size. Default is 3.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples of Z matrix")
    parser.add_argument("--epochs", type=int, default=50,                
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-6,
                        help="Coefficient of weight decay.")
    parser.add_argument("--batch_size", type=float, default=64,
                        help="Batch size.")
    parser.add_argument("--max_width", type=int, default=64,
                        help="Dimension of hidden representation.")
    return parser.parse_known_args()[0]


args = argument_parser()

transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

device = get_device(args)
model = AdaptiveConvNet(input_channels=1,
                       num_classes=10,
                       num_channels=args.max_width,
                       kernel_size=args.kernel,
                       args=args,
                       device=device).to(device)
model = model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.l2)

if not os.path.exists("results"):
    os.mkdir("results")

def evaluate(test_loader):
    loglike = 0
    error_sum = 0

    with torch.no_grad():
        model.eval()
        for i, (data, labels) in enumerate(test_loader):
            data = data.float().to(device)
            labels = labels.long().to(device)
            output = model(data, args.num_samples)
            pred = output.mean(0)
            logits = F.softmax(pred, dim=1)

            ll = -F.nll_loss(logits, labels, reduction="sum").item()
            loglike += ll
            predicted = torch.argmax(logits, 1)
            error = predicted.ne(labels).sum().item()
            error_sum += error

    test_loglikes = loglike / len(test_dataset)
    test_err = error_sum / len(test_dataset)

    test_metrics = {'test_err': round(test_err * 100, 3),
                    'test_like': round(test_loglikes, 3)}

    return test_metrics


train_losses = []
with tqdm(range(args.epochs)) as tq:
    for epoch in tq:
        train_loss = 0.0

        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.float().to(device)
            labels = labels.long().to(device)

            # making grad zero
            optimizer.zero_grad()

            # sample an architecture
            act_vec = model(data, args.num_samples)
            loss = model.estimate_ELBO(loss_fn, act_vec, labels, N_train=len(train_dataset), kl_weight=1)

            loss.backward()
            optimizer.step()

            # adding losses
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        test_results = evaluate(test_loader)
        print("Test error: {} Test Log likelihood: {}".format(test_results['test_err'], test_results['test_like']))

        kl_beta = model.structure_sampler.get_kl()
        tq.set_postfix({'Tr. loss': '%.6f' % train_loss, 'KL Beta': '%.6f' % kl_beta})
    torch.save(model, "results/model_MNIST.pt")

