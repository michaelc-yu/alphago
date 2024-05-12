
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import go
import extract_features
import load_data
import policy
import value
import sys
import argparse



sgffiles = "./sgffiles"

policy_data, value_data = load_data.get_data(sgffiles)


def train_policy(samples = None):
    print(f"Training policy network with {samples} samples")

    policyNN = policy.PolicyNetwork(input_channels=28, k=48)
    # policyNN.load_state_dict(torch.load('policy_network_model.pth'))

    feats, moves = zip(*policy_data)

    if samples is not None:
        feats, moves = feats[:samples], moves[:samples]

    print(f"{len(feats)} total board positions")

    features_tensor = torch.tensor(feats, dtype=torch.float32) # this is extremely slow
    moves_tensor = torch.tensor(moves, dtype=torch.long)
    # print(f"feats: {features_tensor.shape}") # n x 28 x 19 x 19  (# training examples, 28 features each of size 19x19 for each example)
    # print(f"moves: {moves_tensor.shape}") # n x 2  (# training examples, a pair of x,y coordinate for move taken for each example)
    labels_indices = moves_tensor[:, 0] * 19 + moves_tensor[:, 1]
    # print(f"labels: {labels_indices}")

    dataset = TensorDataset(features_tensor, labels_indices)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policyNN.parameters(), lr=0.001)


    for epoch in range(20):
        total_loss = 0
        i = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = policyNN(batch_features)
            # print(f"outputs shape: {outputs.shape}") # [n, 361]  (# training examples, 19 x 19 probabilities)
            # print(f"labels shape: {labels_indices.shape}") # [n]  (# training examples, each one a value 0-360 representing the flattened board to place a move)
            # print(f"outputs: {outputs}")
            # print(f"labels: {labels_indices}")
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            i+=1
            total_loss += loss.item()
        print(f"Epoch {epoch}, Avg Loss: {total_loss / i}")

    # torch.save(policyNN.state_dict(), 'policy_network_model.pth')



def train_value(samples = None):
    print(f"Training value network with {samples} samples")

    valueNN = value.ValueNetwork(input_channels=28, k=48)
    # valueNN.load_state_dict(torch.load('value_network_model.pth'))

    feats, winners = zip(*value_data)

    if samples is not None:
        feats, winners = feats[:samples], winners[:samples]

    print(f"{len(feats)} total board positions")

    features_tensor = torch.tensor(feats, dtype=torch.float32)
    winners_tensor = torch.tensor(winners, dtype=torch.float32)
    winners_tensor = (winners_tensor + 1) / 2
    # winners_tensor = torch.tensor(winners, dtype=torch.float32).unsqueeze(1)
    # print(f"feats: {features_tensor.shape}") # num training examples x 28 x 19 x 19  (# training examples, 28 features each of size 19x19 for each example)
    # print(f"winner: {winners_tensor.shape}") # num training examples (x number of winners where x = num training examples)
    # print(f"winners tensor: {winners_tensor}")

    dataset = TensorDataset(features_tensor, winners_tensor)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(valueNN.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    for epoch in range(20):
        total_loss = 0
        i = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = valueNN(batch_features)
            # print(f"outputs shape: {outputs.shape}") # [num training examples, 1]  (# training examples, 1 winner)
            # print(f"labels shape: {winners_tensor.shape}") # [num training examples]  (# training examples of winners)
            # print(f"outputs: {outputs}") # 1 for black win, -1 for white win
            # print(f"labels: {winners_tensor}")
            # print(f"batch labels: {batch_labels}")
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            i+=1
            total_loss += loss.item()
        print(f"Epoch {epoch}, Avg Loss: {total_loss / i}")

    # torch.save(valueNN.state_dict(), 'value_network_model.pth')


def valid_data_amount(value):
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError("data_amount must be an integer or 'all'.")

def main():
    parser = argparse.ArgumentParser(description="Train neural networks for a game.")

    parser.add_argument("--policy", action="store_true", help="Train the policy network")
    parser.add_argument("--value", action="store_true", help="Train the value network")
    parser.add_argument("--samples", type=valid_data_amount, help="Amount of data to use for training")

    args = parser.parse_args()

    if not args.policy and not args.value:
        parser.error("At least one of --policy or --value must be specified.")

    if args.policy:
        train_policy(args.samples)

    if args.value:
        train_value(args.samples)

if __name__ == "__main__":
    main()

