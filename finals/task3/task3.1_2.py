import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib

# Set the model types
model_types = ["neural_network", "random_forest"]

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def load_models(model_type, script_dir):
    models = []
    for joint_idx in range(7):
        if model_type == "neural_network":
            model = MLP()
            model.load_state_dict(torch.load(os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')))
            model.eval()
        elif model_type == "random_forest":
            model = joblib.load(os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib'))
        models.append(model)
    return models

def generate_predictions(models, model_type, test_input, time_step, joint_vel_limits):
    predicted_joint_positions_over_time = np.zeros((len(test_input), 7))
    for joint_idx in range(7):
        if model_type == "neural_network":
            test_input_tensor = torch.from_numpy(test_input).float()
            with torch.no_grad():
                predictions = models[joint_idx](test_input_tensor).numpy().flatten()
        elif model_type == "random_forest":
            predictions = models[joint_idx].predict(test_input)
        predicted_joint_positions_over_time[:, joint_idx] = predictions
    qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
    return predicted_joint_positions_over_time, np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time_array = np.array(data['time'])

    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    number_of_goal_positions_to_test = 10
    goal_positions = [[
        np.random.uniform(*goal_position_bounds['x']),
        np.random.uniform(*goal_position_bounds['y']),
        np.random.uniform(*goal_position_bounds['z'])
    ] for _ in range(number_of_goal_positions_to_test)]

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    
    # Convert active joints list to a 2D numpy array as required by PinWrapper
    active_joints_2d = np.array([sim.getNameActiveJoints()])
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", active_joints_2d, ["pybullet"], False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    controlled_frame_name = "panda_link8"
    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    all_positions = {model_type: [] for model_type in model_types}
    all_velocities = {model_type: [] for model_type in model_types}

    for model_type in model_types:
        models = load_models(model_type, script_dir)
        
        for goal_position in goal_positions:
            test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
            test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

            positions, velocities = generate_predictions(models, model_type, test_input, time_step, joint_vel_limits)
            all_positions[model_type].append(positions)
            all_velocities[model_type].append(velocities)

    fig, axs = plt.subplots(7, 2, figsize=(16, 24))
    fig.suptitle("Comparison of Neural Network and Random Forest Model Predictions", fontsize=16)

    for joint_idx in range(7):
        for i, goal_position in enumerate(goal_positions):
            axs[joint_idx, 0].plot(test_time_array, all_positions["neural_network"][i][:, joint_idx], label=f'NN Goal {i+1}', linestyle='-', linewidth=0.8)
            axs[joint_idx, 0].plot(test_time_array, all_positions["random_forest"][i][:, joint_idx], label=f'RF Goal {i+1}', linestyle='--', linewidth=0.8)
            
            axs[joint_idx, 1].plot(test_time_array, all_velocities["neural_network"][i][:, joint_idx], label=f'NN Goal {i+1}', linestyle='-', linewidth=0.8)
            axs[joint_idx, 1].plot(test_time_array, all_velocities["random_forest"][i][:, joint_idx], label=f'RF Goal {i+1}', linestyle='--', linewidth=0.8)

        axs[joint_idx, 0].set_title(f'Joint {joint_idx+1} Position')
        axs[joint_idx, 1].set_title(f'Joint {joint_idx+1} Velocity')
        axs[joint_idx, 0].set_xlabel("Time (s)")
        axs[joint_idx, 1].set_xlabel("Time (s)")
        axs[joint_idx, 0].set_ylabel("Position")
        axs[joint_idx, 1].set_ylabel("Velocity")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 0.95), fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
