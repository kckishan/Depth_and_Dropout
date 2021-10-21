#!/usr/bin/env python
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import numpy as np

from src.models.MLP import AdaptiveMLP
from src.dataloader.toy_dataset import generate_1d_regression
from src.dataloader.loader import DatasetLoader
from src.utils import table_printer, get_device

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists("models"):
    os.mkdir("models")


def argument_parser():
    parser = argparse.ArgumentParser(description="Run Nonparametric Bayesian Architecture Learning")
    parser.add_argument("--prior_temp", type=float, default=1.,
                        help="Temperature for prior Concrete Bernoulli ")
    parser.add_argument("--temp", type=float, default=.1,
                        help="Temperature for posterior Concrete Bernoulli")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon to select the activated layers")
    parser.add_argument("--truncation_level", type=int, default=100,
                        help="K: Truncation level for variational approximation")
    parser.add_argument('--use_cuda', action='store_false',
                        help="Use CPU or GPU")
    parser.add_argument("--a_prior", type=float, default=1.1,
                        help="a parameter for Beta distribution")
    parser.add_argument("--b_prior", type=float, default=2.,
                        help="b parameter for Beta distribution")
    parser.add_argument("--validation-percentage", type=int, default=.2,
                        help="Percentage of validation samples.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples of Z matrix")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Coefficient of weight decay.")
    parser.add_argument("--batch_size", type=float, default=200,
                        help="Batch size.")
    parser.add_argument("--max_width", type=int, default=50,
                        help="Number of neurons in hidden layer.")
    parser.add_argument("--ntrain", type=int, default=500,
                        help="Number of training data points to generate")
    parser.add_argument("--id", type=int, default=0,
                        help="run identifier")

    return parser.parse_known_args()[0]


args = argument_parser()
run_id = f'S{args.num_samples}_a{args.a_prior}_b{args.b_prior}_{args.id}'
table_printer(args)

# Define the settings to generate synthetic data
domain = (-2.0, 2.0)
noise_std = 0.1

# sample `ntrain` data points for training
xs, ys = generate_1d_regression(
    n_points=args.ntrain,
    domain=domain,
    noise_std=noise_std,
    seed=7
)
X_train = xs.reshape(-1, 1)
Y_train = ys.reshape(-1, 1)

# sample 700 data points for testing
xs_t, ys_t = generate_1d_regression(
    n_points=700,
    domain=domain,
    noise_std=noise_std,
    seed=None
)
X_test_t = torch.from_numpy(xs_t)
Y_test_t = torch.from_numpy(ys_t)

# plot synthetic data
plt.figure(figsize=(4, 3))
plt.scatter(X_train.ravel(), Y_train.ravel(), s=0.5, c='black')
plt.show()

# Define dataset and data loader for training
train_dataset = DatasetLoader(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

device = get_device(args)
model = AdaptiveMLP(input_feature_dim=1, out_feature_dim=1, args=args, device=device)
model = model.to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

train_losses = []
best_rmse = float('Inf')
a_k = []
b_k = []
with tqdm(range(args.epochs)) as tq:
    for epoch in tq:
        train_loss = 0.0

        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data = data.float().to(device)
            labels = labels.float().to(device)

            # making grad zero
            optimizer.zero_grad()

            # sample an architecture
            act_vec = model(data, args.num_samples)
            loss = model.estimate_ELBO(loss_fn, act_vec, labels, N_train=len(train_dataset), kl_weight=1)

            loss.backward()
            optimizer.step()

            # adding losses
            train_loss += loss.item()

        beta_a, beta_b = model.structure_sampler.get_variational_params()
        a_k.append(beta_a.cpu().detach().numpy())
        b_k.append(beta_b.cpu().detach().numpy())
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        kl_loss = model.structure_sampler.get_kl().item()
        tq.set_postfix(
            {'Tr. loss': '%.6f' % train_loss, 'KL Beta': '%.6f' % kl_loss})

with torch.no_grad():
    model.eval()

    inputs = X_test_t.to(device)
    preds = model(inputs, num_samples=10)

pred_mu = preds.mean(0)
test_loss = loss_fn(pred_mu, Y_test_t.to(device))
test_rmse = (test_loss.mean() ** 0.5).item()

print("RMSE:", test_rmse)
result = {'rmse': test_rmse}
torch.save(result, f'results/{run_id}.pt')

alpha_beta = {'alpha': a_k,
              'beta': b_k}
torch.save(alpha_beta, f'alpha_beta_{run_id}.pt')


# plot training loss
plt.figure(figsize=(4, 3))
plt.plot(np.arange(len(train_losses)), train_losses, c='blue')
plt.show()