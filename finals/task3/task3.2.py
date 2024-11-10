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
    controlled_frame_name = "panda_link8"
    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    all_results = {
        "neural_network": {"q_des": [], "q_mes": [], "tau_cmd": []},
        "random_forest": {"q_des": [], "q_mes": [], "tau_cmd": []}
    }

    for model_type, models in [("neural_network", nn_models), ("random_forest", rf_models)]:
        print(f"Running simulation with {model_type} model")
        
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

        predicted_joint_positions = np.zeros((len(test_time_array), 7))
        for joint_idx in range(7):
            if model_type == "neural_network":
                with torch.no_grad():
                    predicted_joint_positions[:, joint_idx] = models[joint_idx](torch.from_numpy(test_input).float()).numpy().flatten()
            elif model_type == "random_forest":
                predicted_joint_positions[:, joint_idx] = models[joint_idx].predict(test_input)

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
        all_results[model_type]["q_des"] = np.array(q_des_list[:max_len])
        all_results[model_type]["q_mes"] = np.array(q_mes_list[:max_len])
        all_results[model_type]["tau_cmd"] = np.array(tau_cmd_list[:max_len])

        # Plot tracking error for each joint
    fig1, axs1 = plt.subplots(7, 1, figsize=(10, 14))
    fig1.suptitle("Tracking Error for MLP and Random Forest Models", fontsize=12)

    for joint_idx in range(7):
        nn_tracking_error = all_results["neural_network"]["q_des"][:, joint_idx] - all_results["neural_network"]["q_mes"][:, joint_idx]
        rf_tracking_error = all_results["random_forest"]["q_des"][:, joint_idx] - all_results["random_forest"]["q_mes"][:, joint_idx]

        axs1[joint_idx].plot(test_time_array[:max_len], nn_tracking_error, label='MLP Tracking Error', color='blue', linewidth=0.8)
        axs1[joint_idx].plot(test_time_array[:max_len], rf_tracking_error, label='Random Forest Tracking Error', color='green', linestyle='--', linewidth=0.8)
        axs1[joint_idx].set_title(f"Joint {joint_idx + 1} Tracking Error", fontsize=8)
        axs1[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs1[joint_idx].set_ylabel("Error (rad)", fontsize=6)
        axs1[joint_idx].legend(fontsize=5, loc="upper right")
        axs1[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Plot control torques for each joint
    fig2, axs2 = plt.subplots(7, 1, figsize=(10, 14))
    fig2.suptitle("Control Torques for MLP and Random Forest Models", fontsize=12)

    for joint_idx in range(7):
        nn_tau_cmd = all_results["neural_network"]["tau_cmd"][:, joint_idx]
        rf_tau_cmd = all_results["random_forest"]["tau_cmd"][:, joint_idx]

        axs2[joint_idx].plot(test_time_array[:max_len], nn_tau_cmd, label='MLP Control Torque', color='blue', linewidth=0.8)
        axs2[joint_idx].plot(test_time_array[:max_len], rf_tau_cmd, label='Random Forest Control Torque', color='green', linestyle='--', linewidth=0.8)
        axs2[joint_idx].set_title(f"Joint {joint_idx + 1} Control Torque", fontsize=8)
        axs2[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs2[joint_idx].set_ylabel("Torque (Nm)", fontsize=6)
        axs2[joint_idx].legend(fontsize=5, loc="upper right")
        axs2[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Plot absolute tracking error for each joint
    fig3, axs3 = plt.subplots(7, 1, figsize=(10, 14))
    fig3.suptitle("Absolute Tracking Error for MLP and Random Forest Models", fontsize=12)

    for joint_idx in range(7):
        nn_tracking_error = all_results["neural_network"]["q_des"][:, joint_idx] - all_results["neural_network"]["q_mes"][:, joint_idx]
        rf_tracking_error = all_results["random_forest"]["q_des"][:, joint_idx] - all_results["random_forest"]["q_mes"][:, joint_idx]

        axs3[joint_idx].plot(test_time_array[:max_len], np.abs(nn_tracking_error), label="MLP Abs Error", color="blue", alpha=0.5, linewidth=0.8)
        axs3[joint_idx].plot(test_time_array[:max_len], np.abs(rf_tracking_error), label="Random Forest Abs Error", color="green", linestyle="--", alpha=0.5, linewidth=0.8)
        axs3[joint_idx].set_title(f"Joint {joint_idx + 1} Absolute Tracking Error", fontsize=8)
        axs3[joint_idx].set_xlabel("Time (s)", fontsize=6)
        axs3[joint_idx].set_ylabel("Abs Error (rad)", fontsize=6)
        axs3[joint_idx].legend(fontsize=5, loc="upper right")
        axs3[joint_idx].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == '__main__':
    main()
