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
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define MLP Model with variable hidden nodes
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

# Loop over different hidden node sizes
hidden_node_list = [32, 64, 96, 128]
results = {}

for hidden_nodes in hidden_node_list:
    print(f'\nTraining with hidden nodes: {hidden_nodes}')
    
    # Initialize model, loss function, and optimizer
    model = ShallowCorrectorMLP(hidden_nodes=hidden_nodes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    # Train model
    train_losses = []
    epochs = 1000
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:  # Print every 100 epochs
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    # Store the loss curve for each hidden node size
    results[hidden_nodes] = train_losses

# Plot training loss for different hidden node sizes
plt.figure(figsize=(12, 6))
for hidden_nodes, losses in results.items():
    plt.plot(losses, label=f'Hidden Nodes: {hidden_nodes}')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss for Different Hidden Nodes in MLP')
plt.legend()
plt.show()

# Testing Phase: Simulate trajectory tracking with and without MLP correction
q_test = 0
dot_q_test = 0
q_real = []
q_real_corrected = []

# Integration with only PD Control (no MLP correction)
for i in range(len(t)):
    tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
    ddot_q_real = (tau - b * dot_q_test) / m
    dot_q_test += ddot_q_real * dt
    q_test += dot_q_test * dt
    q_real.append(q_test)

# Reset initial conditions for corrected simulation
q_test = 0
dot_q_test = 0

# Use the model with the last hidden_nodes size (128) for correction
for i in range(len(t)):
    tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
    inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
    correction = model(inputs.unsqueeze(0)).item()
    ddot_q_corrected = (tau - b * dot_q_test + correction) / m
    dot_q_test += ddot_q_corrected * dt
    q_test += dot_q_test * dt
    q_real_corrected.append(q_test)

# Plot results for trajectory tracking
plt.figure(figsize=(12, 6))
plt.plot(t, q_target, 'r-', label='Target')
plt.plot(t, q_real, 'b--', label='PD Only')
plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
plt.title('Trajectory Tracking with and without MLP Correction')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.legend()
plt.show()
