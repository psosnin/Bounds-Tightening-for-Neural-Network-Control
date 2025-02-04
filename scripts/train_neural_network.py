import sys
import argparse

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn

sys.path.append("..")  # nopep8
from src import network
from configs import *

"""
Get command line arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int)
parser.add_argument("--network", "-n", type=int)
args = parser.parse_args()

seed = args.seed
network_type = args.network
shape = SHAPES[network_type]

"""
Set up configuration, wandb run and dataset
"""

np.random.seed(seed)
torch.manual_seed(seed)

run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    job_type=TRAINING_JOB_TYPE,
    group=GROUP.format(seed=seed),
    name=TRAINING_JOB_NAME.format(network_type=network_type, seed=seed),
    mode=MODE,
    config={
        "seed": seed,
        "num_epochs": NUM_EPOCHS,  # number of epochs
        "shape": str(shape),  # network shape
        "network_type": network_type,  # network type
        "batchsize": BATCHSIZE,  # batch size
        "learning_rate": LEARNING_RATE,  # learning rate
    },
)
config = wandb.config
device = torch.device('cpu')

dataset = run.use_artifact(DATASET_ARTIFACT_NAME.format(seed=seed) + ":latest")
config.system = dataset.metadata["system"]
dataset_dir = dataset.download()
x = np.load(f"{dataset_dir}/{X_FILENAME}").astype(np.float32)
u = np.load(f"{dataset_dir}/{U_FILENAME}").astype(np.float32)

input_dim = x.shape[1]
output_dim = u.shape[1]

dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(u))
n_train = int(0.9 * len(dataset))
train_set, test_set = random_split(dataset, [n_train, len(dataset) - n_train])
train_loader = DataLoader(train_set, batch_size=config.batchsize, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

"""
Set up neural network model and train
"""

shape = [input_dim] + shape + [output_dim]
model = network.NeuralNet(shape).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

n_total_steps = len(train_loader)
for epoch in range(config.num_epochs):
    for i, (x, u) in enumerate(train_loader):
        u = u.to(device)
        x = x.to(device)

        # Forward pass
        output = model(x)
        loss = criterion(output.squeeze(), u.squeeze())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"MSE": loss.item()})

"""
Test the model
"""

with torch.no_grad():
    sq_error = 0
    for x_test, u_test in test_loader:
        u_test = u_test.to(device)
        x_test = x_test.to(device).squeeze(1)
        outputs = model(x_test)
        sq_error += torch.sum((outputs - u_test) ** 2)
    acc = sq_error / len(test_set)
    wandb.log({"Test MSE": acc.item()})

"""
Save network as an artifact
"""

artifact = wandb.Artifact(
    name=NETWORK_ARTIFACT_NAME.format(network_type=network_type, seed=seed),
    type=NETWORK_ARTIFACT_TYPE,
    metadata=dict(config)
)

with artifact.new_file(MODEL_FILENAME, mode="wb") as file:
    torch.save(model.state_dict(), file)

with artifact.new_file(PARAMS_FILENAME, mode="wb") as file:
    params = [param.cpu().detach().numpy() for param in list(model.parameters())]
    params = list(zip(params[::2], params[1::2]))
    np.save(file, np.array(params, dtype=object), allow_pickle=True)

run.log_artifact(artifact)
