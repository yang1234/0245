import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl

# Define MLP Model
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

def load_models(script_dir, model_type):
    models = []
    if model_type == "neural_network":
        for joint_idx in range(7):
            model = MLP()
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif model_type == "random_forest":
        for joint_idx in range(7):
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        raise ValueError("Invalid model type specified.")
    return models

# Exponential Moving Average filter
def exponential_moving_average(data, alpha=0.1):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

def run_simulation(model_type, models, sim, dyn_model, goal_position, test_time_array, time_step, joint_vel_limits, smooth=False):
    test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
    test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

    # Predict joint positions
    predicted_joint_positions = np.zeros((len(test_time_array), 7))
    for joint_idx in range(7):
        if model_type == "neural_network":
            with torch.no_grad():
                predicted_joint_positions[:, joint_idx] = models[joint_idx](torch.from_numpy(test_input).float()).numpy().flatten()
        elif model_type == "random_forest":
            predictions = models[joint_idx].predict(test_input)
            if smooth:
                predictions = exponential_moving_average(predictions)
            predicted_joint_positions[:, joint_idx] = predictions

    qd_des = np.gradient(predicted_joint_positions, axis=0, edge_order=2) / time_step
    qd_des_clipped = np.clip(qd_des, -np.array(joint_vel_limits), np.array(joint_vel_limits))

    sim.ResetPose()
    cmd = MotorCommands()
    kp = 1000
    kd = 100
    current_time = 0

    q_des_list = []
    q_mes_list = []
    tau_cmd_list = []

    while current_time < test_time_array.max():
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        
        current_index = min(int(current_time / time_step), len(test_time_array) - 1)
        q_des = predicted_joint_positions[current_index, :]
        qd_des_clip = qd_des_clipped[current_index, :]

        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
        sim.Step(cmd, "torque")

        q_des_list.append(q_des)
        q_mes_list.append(q_mes)
        tau_cmd_list.append(tau_cmd)

        current_time += time_step

    max_len = min(len(test_time_array), len(q_des_list))
    return {
        "q_des": np.array(q_des_list[:max_len]),
        "q_mes": np.array(q_mes_list[:max_len]),
        "tau_cmd": np.array(tau_cmd_list[:max_len]),
        "predicted_joint_positions": predicted_joint_positions
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time_array = np.array(data['time'])

    goal_position_bounds = {'x': (0.6, 0.8), 'y': (-0.1, 0.1), 'z': (0.12, 0.12)}
    goal_position = [
        np.random.uniform(*goal_position_bounds['x']),
        np.random.uniform(*goal_position_bounds['y']),
        np.random.uniform(*goal_position_bounds['z'])
    ]

    nn_models = load_models(script_dir, "neural_network")
    rf_models = load_models(script_dir, "random_forest")

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    dyn_model = PinWrapper(conf_file_name, "pybullet", np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0), ["pybullet"], False, 0, root_dir)
    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    # Run simulation for each model and collect results
    results_nn = run_simulation("neural_network", nn_models, sim, dyn_model, goal_position, test_time_array, time_step, joint_vel_limits)
    results_rf = run_simulation("random_forest", rf_models, sim, dyn_model, goal_position, test_time_array, time_step, joint_vel_limits)
    results_rf_smooth = run_simulation("random_forest", rf_models, sim, dyn_model, goal_position, test_time_array, time_step, joint_vel_limits, smooth=True)

    import matplotlib.pyplot as plt

    # Plot Joint Positions for MLP, RF, and RF Smoothed
    fig_pos, axs_pos = plt.subplots(7, 1, figsize=(10, 14))
    fig_pos.suptitle("Joint Positions Comparison for MLP, RF, and RF Smoothed", fontsize=12)

    for joint_idx in range(7):
        axs_pos[joint_idx].plot(test_time_array[:len(results_nn["q_des"])], results_nn["q_des"][:, joint_idx], label="MLP q_des", color="blue", linewidth=0.8)
        axs_pos[joint_idx].plot(test_time_array[:len(results_rf["q_des"])], results_rf["q_des"][:, joint_idx], label="RF q_des", color="green", linewidth=0.8)
        axs_pos[joint_idx].plot(test_time_array[:len(results_rf_smooth["q_des"])], results_rf_smooth["q_des"][:, joint_idx], label="RF Smooth q_des", color="orange", linewidth=0.8)
        axs_pos[joint_idx].set_title(f"Joint {joint_idx + 1} Position", fontsize=8)
        axs_pos[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs_pos[joint_idx].set_ylabel("Position (rad)", fontsize=6)
        axs_pos[joint_idx].tick_params(axis='both', labelsize=6)
        axs_pos[joint_idx].legend(fontsize=5, loc="upper right")
        axs_pos[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Plot Tracking Errors for MLP, RF, and RF Smoothed
    fig_error, axs_error = plt.subplots(7, 1, figsize=(10, 14))
    fig_error.suptitle("Tracking Error Comparison for MLP, RF, and RF Smoothed", fontsize=12)

    for joint_idx in range(7):
        tracking_error_nn = results_nn["q_des"][:, joint_idx] - results_nn["q_mes"][:, joint_idx]
        tracking_error_rf = results_rf["q_des"][:, joint_idx] - results_rf["q_mes"][:, joint_idx]
        tracking_error_rf_smooth = results_rf_smooth["q_des"][:, joint_idx] - results_rf_smooth["q_mes"][:, joint_idx]

        axs_error[joint_idx].plot(test_time_array[:len(tracking_error_nn)], tracking_error_nn, label="MLP Tracking Error", color="blue", linewidth=0.8)
        axs_error[joint_idx].plot(test_time_array[:len(tracking_error_rf)], tracking_error_rf, label="RF Tracking Error", color="green", linewidth=0.8)
        axs_error[joint_idx].plot(test_time_array[:len(tracking_error_rf_smooth)], tracking_error_rf_smooth, label="RF Smooth Tracking Error", color="orange", linewidth=0.8)
        axs_error[joint_idx].set_title(f"Joint {joint_idx + 1} Tracking Error", fontsize=8)
        axs_error[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs_error[joint_idx].set_ylabel("Error (rad)", fontsize=6)
        axs_error[joint_idx].tick_params(axis='both', labelsize=6)
        axs_error[joint_idx].legend(fontsize=5, loc="upper right")
        axs_error[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Plot Control Torques for MLP, RF, and RF Smoothed
    fig_torque, axs_torque = plt.subplots(7, 1, figsize=(10, 14))
    fig_torque.suptitle("Control Torque Comparison for MLP, RF, and RF Smoothed", fontsize=12)

    for joint_idx in range(7):
        axs_torque[joint_idx].plot(test_time_array[:len(results_nn["tau_cmd"])], results_nn["tau_cmd"][:, joint_idx], label="MLP Tau", color="blue", linewidth=0.8)
        axs_torque[joint_idx].plot(test_time_array[:len(results_rf["tau_cmd"])], results_rf["tau_cmd"][:, joint_idx], label="RF Tau", color="green", linewidth=0.8)
        axs_torque[joint_idx].plot(test_time_array[:len(results_rf_smooth["tau_cmd"])], results_rf_smooth["tau_cmd"][:, joint_idx], label="RF Smooth Tau", color="orange", linewidth=0.8)
        axs_torque[joint_idx].set_title(f"Joint {joint_idx + 1} Control Torque", fontsize=8)
        axs_torque[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs_torque[joint_idx].set_ylabel("Torque (Nm)", fontsize=6)
        axs_torque[joint_idx].tick_params(axis='both', labelsize=6)
        axs_torque[joint_idx].legend(fontsize=5, loc="upper right")
        axs_torque[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()



if __name__ == '__main__':
    main()
