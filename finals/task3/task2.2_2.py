import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "neural_network" to use Neural Network models

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

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])  # Shape: (N,)

    # Load all the models in a list
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
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    goal_positions = [
        [
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ]
    ]

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))

    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)

    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    print(f"Initial joint angles: {init_joint_angles}")

    cmd = MotorCommands()
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain
    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    tracking_errors = []
    control_torques = []

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        sim.ResetPose()
        current_time = 0

        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))

        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                test_input_tensor = torch.from_numpy(test_input).float()
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()
            elif neural_network_or_random_forest == "random_forest":
                predictions = models[joint_idx].predict(test_input)
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

        while current_time < test_time_array.max():
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            q_des = predicted_joint_positions_over_time[current_index, :]
            qd_des_clip = qd_des_over_time_clipped[current_index, :]

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")

            tracking_error = np.linalg.norm(q_mes - q_des)
            tracking_errors.append(tracking_error)
            control_torques.append(tau_cmd)

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            time.sleep(time_step)
            current_time += time_step

        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tracking_errors[5:4999])
    plt.title('Tracking Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (Norm)')
    plt.subplot(2, 1, 2)
    plt.plot(control_torques[5:4999])
    plt.title('Control Torques Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Torque')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
