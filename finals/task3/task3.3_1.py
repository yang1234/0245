import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "neural_network" for testing neural network models

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def smooth_trajectory(trajectory, alpha=0.1):
    """
    Apply exponential moving average (EMA) to smooth the trajectory.
    :param trajectory: numpy array of shape (num_samples, num_joints)
    :param alpha: smoothing factor
    :return: smoothed trajectory
    """
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0] = trajectory[0]  # Initialize with the first value
    for i in range(1, len(trajectory)):
        smoothed_trajectory[i] = alpha * trajectory[i] + (1 - alpha) * smoothed_trajectory[i-1]
    return smoothed_trajectory

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract time array and normalize if needed
    time_array = np.array(data['time'])

    # Load all models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            model = MLP()
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)

    # Generate a test goal position
    goal_position = [0.7, 0.0, 0.12]
    test_time_array = np.arange(time_array.min(), time_array.max(), 0.02)  # Simulated time steps
    test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
    test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

    # Predict joint positions over time for the goal
    predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))
    for joint_idx in range(7):
        if neural_network_or_random_forest == "neural_network":
            test_input_tensor = torch.from_numpy(test_input).float()
            with torch.no_grad():
                predictions = models[joint_idx](test_input_tensor).numpy().flatten()
        elif neural_network_or_random_forest == "random_forest":
            predictions = models[joint_idx].predict(test_input)
        predicted_joint_positions_over_time[:, joint_idx] = predictions

    # Smooth the predicted joint positions (apply EMA)
    smoothed_joint_positions_over_time = smooth_trajectory(predicted_joint_positions_over_time, alpha=0.1)

    # Compute the joint velocities (qd_des_over_time) by differentiating the smoothed predicted positions
    time_step = test_time_array[1] - test_time_array[0]
    qd_des_over_time = np.gradient(smoothed_joint_positions_over_time, axis=0, edge_order=2) / time_step

    # Clip the joint velocities to joint limits (for simplicity, assume arbitrary limits)
    joint_vel_limits = np.array([2.0] * 7)  # Example velocity limits for each joint (in radians per second)
    qd_des_over_time_clipped = np.clip(qd_des_over_time, -joint_vel_limits, joint_vel_limits)

    # Initialize simulation and control objects
    sim = pb.SimInterface("pandaconfig.json", conf_file_path_ext=script_dir)
    dyn_model = PinWrapper("pandaconfig.json", "pybullet", np.array(sim.getNameActiveJoints()).reshape(1, -1), ["pybullet"], False, 0, script_dir)
    cmd = MotorCommands()

    # PD controller gains
    kp = 1000
    kd = 100

    # Data collection loop for tracking error analysis
    tracking_errors = []  # To store tracking errors for each time step
    control_inputs = []  # To store control inputs (torque commands)

    for i in range(len(test_time_array)):
        # Current state of the robot in the simulation
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)

        # Get the desired joint positions and velocities
        q_des = smoothed_joint_positions_over_time[i, :]
        qd_des_clip = qd_des_over_time_clipped[i, :]

        # Calculate the tracking error for each joint
        tracking_error = np.linalg.norm(q_des - q_mes)
        tracking_errors.append(tracking_error)

        # Compute the control input (torque command) using feedback linearization
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
        control_inputs.append(tau_cmd)

        # Apply the torque command to the simulation
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
        sim.Step(cmd, "torque")

        time.sleep(time_step)  # Simulation step time

    # Plot tracking errors over time
    plt.figure(figsize=(10, 6))
    plt.plot(test_time_array, tracking_errors, label="Tracking Error (Smoothed)", color="red")
    plt.title("Tracking Error Over Time (Smoothed Trajectory)")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error (Norm)")
    plt.legend()
    plt.show()

    # Plot control inputs (torque commands) for smoothed trajectory
    plt.figure(figsize=(10, 6))
    for i in range(7):
        control_torque = [tau[i] for tau in control_inputs]
        plt.plot(test_time_array, control_torque, label=f"Joint {i+1} (Smoothed)")
    plt.title("Control Torque Commands Over Time (Smoothed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Control Torque [Nm]")
    plt.legend()
    plt.show()

    # Plot original (unfiltered) predicted joint trajectories
    plt.figure(figsize=(12, 8))
    for i in range(7):
        plt.subplot(3, 3, i + 1)
        plt.plot(predicted_joint_positions_over_time[:, i], label=f"Joint {i+1} (Original)", color='gray')
        plt.title(f"Joint {i+1} Predicted Trajectory (Original)")
        plt.xlabel("Time Steps")
        plt.ylabel("Joint Position")
        plt.legend()
    plt.suptitle("Original Joint Trajectories (No Smoothing)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot smoothed joint trajectories
    plt.figure(figsize=(12, 8))
    for i in range(7):
        plt.subplot(3, 3, i + 1)
        plt.plot(smoothed_joint_positions_over_time[:, i], label=f"Joint {i+1} (Smoothed)", color='green')
        plt.plot(predicted_joint_positions_over_time[:, i], label=f"Joint {i+1} (Original)", color='gray')
        plt.title(f"Joint {i+1} Predicted Trajectory (Smoothed)")
        plt.xlabel("Time Steps")
        plt.ylabel("Joint Position")
        plt.legend()
    plt.suptitle("Smoothed Joint Trajectories")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()
