import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
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
    # PD control output
    tau = k_p * (q_target_train[i] - q) + k_d * (dot_q_target_train[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X_train.append([q, dot_q, q_target_train[i], dot_q_target_train[i]])
    Y_train.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Generate testing data
q = 0
dot_q = 0
X_test = []
Y_test = []

for i in range(num_test_samples):
    # PD control output
    tau = k_p * (q_target_test[i] - q) + k_d * (dot_q_target_test[i] - dot_q)
    # Ideal motor dynamics
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X_test.append([q, dot_q, q_target_test[i], dot_q_target_test[i]])
    Y_test.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader for training
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Dataset and DataLoader for testing
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define shallow MLP model (1 hidden layer)
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, hidden_nodes=64):
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
    def __init__(self, hidden_nodes=64):
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

# Training function
def train_model(model, train_loader, lr, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Learning Rate {lr}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    return train_losses

# Testing function
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

# Learning rates to test
learning_rates = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
hidden_nodes = 64
results_train_loss = {}
results_test_loss = {}

# Train and evaluate models with different learning rates
for lr in learning_rates:
    print(f'\nTraining ShallowCorrectorMLP with learning rate: {lr}')
    shallow_model = ShallowCorrectorMLP(hidden_nodes=hidden_nodes)
    train_losses = train_model(shallow_model, train_loader, lr)
    test_loss = test_model(shallow_model, test_loader)
    results_train_loss[(lr, 'Shallow')] = train_losses
    results_test_loss[(lr, 'Shallow')] = test_loss

    print(f'\nTraining DeepCorrectorMLP with learning rate: {lr}')
    deep_model = DeepCorrectorMLP(hidden_nodes=hidden_nodes)
    train_losses = train_model(deep_model, train_loader, lr)
    test_loss = test_model(deep_model, test_loader)
    results_train_loss[(lr, 'Deep')] = train_losses
    results_test_loss[(lr, 'Deep')] = test_loss


    

# Plot training loss for different learning rates, with separate subplots for shallow and deep MLP
plt.figure(figsize=(12, 12))

# Plot for Shallow MLP
plt.subplot(2, 1, 1)
for (lr, model_type), losses in results_train_loss.items():
    if model_type == 'Shallow':
        label = f'Shallow MLP, LR: {lr}'
        plt.plot(losses, label=label)
plt.ylim(0, 1.5)  # Set y-axis range from 0 to 1.5
plt.yticks(np.arange(0, 1.6, 0.05))  # Set y-axis ticks from 0 to 1.5 with interval of 0.05
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss for Shallow MLP with Different Learning Rates')
plt.legend()

# Plot for Deep MLP
plt.subplot(2, 1, 2)
for (lr, model_type), losses in results_train_loss.items():
    if model_type == 'Deep':
        label = f'Deep MLP, LR: {lr}'
        plt.plot(losses, linestyle='--', label=label)
plt.ylim(0, 1.5)  # Set y-axis range from 0 to 1.5
plt.yticks(np.arange(0, 1.6, 0.05))  # Set y-axis ticks from 0 to 1.5 with interval of 0.05
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss for Deep MLP with Different Learning Rates')
plt.legend()

plt.tight_layout()
plt.show()

# Prepare data for plotting
shallow_test_losses = [results_test_loss[(lr, 'Shallow')] for lr in learning_rates]
deep_test_losses = [results_test_loss[(lr, 'Deep')] for lr in learning_rates]

# Plot test losses for different learning rates
plt.figure(figsize=(10, 6))
x = np.arange(len(learning_rates))

# Plot Shallow MLP test losses
plt.plot(x, shallow_test_losses, marker='o', linestyle='-', color='blue', label='Shallow MLP')

# Plot Deep MLP test losses
plt.plot(x, deep_test_losses, marker='o', linestyle='--', color='green', label='Deep MLP')

# Configure plot labels and ticks
plt.xlabel('Learning Rates')
plt.ylabel('Test Loss')
plt.title('Test Loss for Shallow and Deep MLPs with Different Learning Rates')
plt.xticks(x, [str(lr) for lr in learning_rates])  # Set x-axis labels to the learning rates
plt.legend()

plt.tight_layout()
plt.show()

# Print test losses for different learning rates
print("\nTest Losses (Generalization Performance) on Entire Domain:")
for (lr, model_type), test_loss in results_test_loss.items():
    print(f'{model_type} MLP, LR: {lr}, Test Loss: {test_loss:.6f}')
