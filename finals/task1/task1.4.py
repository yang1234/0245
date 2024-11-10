import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in training dataset
num_test_samples = 500  # Number of samples in testing dataset

# Generate synthetic data for trajectory tracking (Training Data)
t_train = np.linspace(0, 10, num_samples)
q_target_train = np.sin(t_train)
dot_q_target_train = np.cos(t_train)

# Generate synthetic data for trajectory tracking (Testing Data)
t_test = np.linspace(10, 15, num_test_samples)  # New time interval for testing
q_target_test = np.sin(t_test)
dot_q_target_test = np.cos(t_test)

# Generate training data
q = 0
dot_q = 0
X_train = []
Y_train = []

for i in range(num_samples):
    tau = k_p * (q_target_train[i] - q) + k_d * (dot_q_target_train[i] - dot_q)
    ddot_q_real = (tau - b * dot_q) / m
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    X_train.append([q, dot_q, q_target_train[i], dot_q_target_train[i]])
    Y_train.append(ddot_q_error)
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Generate testing data
q = 0
dot_q = 0
X_test = []
Y_test = []

for i in range(num_test_samples):
    tau = k_p * (q_target_test[i] - q) + k_d * (dot_q_target_test[i] - dot_q)
    ddot_q_real = (tau - b * dot_q) / m
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    X_test.append([q, dot_q, q_target_test[i], dot_q_target_test[i]])
    Y_test.append(ddot_q_error)
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# Define shallow MLP model (1 hidden layer)
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, hidden_nodes=32):
        super(ShallowCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Define deep MLP model (2 hidden layers)
class DeepCorrectorMLP(nn.Module):
    def __init__(self, hidden_nodes=32):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Train function
def train_model(model, train_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
    
    return train_losses

# Test function to evaluate on the test set
def test_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

# Batch sizes to test
batch_sizes = [64, 128, 256, 1000]
results_shallow = {}
results_deep = {}
results_train_time_shallow = {}
results_train_time_deep = {}

for batch_size in batch_sizes:
    # Create DataLoader with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training ShallowCorrectorMLP
    shallow_model = ShallowCorrectorMLP(hidden_nodes=32)
    start_time = time.time()
    shallow_losses = train_model(shallow_model, train_loader)
    shallow_train_time = time.time() - start_time
    shallow_test_loss = test_model(shallow_model, test_loader)
    results_shallow[batch_size] = (shallow_losses, shallow_test_loss)
    results_train_time_shallow[batch_size] = shallow_train_time
    
    # Training DeepCorrectorMLP
    deep_model = DeepCorrectorMLP(hidden_nodes=32)
    start_time = time.time()
    deep_losses = train_model(deep_model, train_loader)
    deep_train_time = time.time() - start_time
    deep_test_loss = test_model(deep_model, test_loader)
    results_deep[batch_size] = (deep_losses, deep_test_loss)
    results_train_time_deep[batch_size] = deep_train_time

# Plot training loss for different batch sizes
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Plot ShallowCorrectorMLP losses
for batch_size, (losses, _) in results_shallow.items():
    axs[0].plot(losses, label=f'Shallow MLP, Batch Size: {batch_size}')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Training Loss')
axs[0].set_title('Training Loss for Shallow MLP with Different Batch Sizes')
axs[0].legend()

# Plot DeepCorrectorMLP losses
for batch_size, (losses, _) in results_deep.items():
    axs[1].plot(losses, label=f'Deep MLP, Batch Size: {batch_size}', linestyle='--')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Training Loss')
axs[1].set_title('Training Loss for Deep MLP with Different Batch Sizes')
axs[1].legend()

plt.tight_layout()
plt.show()

# Plot test loss for different batch sizes
fig, ax1 = plt.subplots(figsize=(10, 6))
batch_size_labels = [str(bs) for bs in batch_sizes]
shallow_test_losses = [results_shallow[bs][1] for bs in batch_sizes]
deep_test_losses = [results_deep[bs][1] for bs in batch_sizes]

x = np.arange(len(batch_sizes))
width = 0.35  # Bar width

# Test Losses
ax1.bar(x - width/2, shallow_test_losses, width, label='Shallow MLP Test Loss')
ax1.bar(x + width/2, deep_test_losses, width, label='Deep MLP Test Loss')
ax1.set_xticks(x)
ax1.set_xticklabels(batch_size_labels)
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Test Loss')
ax1.set_title('Test Loss for Shallow and Deep MLP with Different Batch Sizes')
ax1.legend()

plt.show()

# Plot training times for different batch sizes
fig, ax2 = plt.subplots(figsize=(10, 6))
shallow_train_times = [results_train_time_shallow[bs] for bs in batch_sizes]
deep_train_times = [results_train_time_deep[bs] for bs in batch_sizes]

# Training Times
ax2.bar(x - width/2, shallow_train_times, width, label='Shallow MLP Training Time')
ax2.bar(x + width/2, deep_train_times, width, label='Deep MLP Training Time')
ax2.set_xticks(x)
ax2.set_xticklabels(batch_size_labels)
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time for Shallow and Deep MLP with Different Batch Sizes')
ax2.legend()

plt.show()
