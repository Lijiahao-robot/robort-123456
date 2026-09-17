import csv
import numpy as np
import matplotlib.pyplot as plt
import os

class DataLogger:
    def __init__(self, save_path="log_data.csv"):
        self.save_path = save_path
        self.fields = ["time","ref_zmp_x","ref_zmp_y","real_zmp_x","real_zmp_y",
                       "ref_com_x","ref_com_y","real_com_x","real_com_y",
                       "joint_hip_l","joint_knee_l","joint_hip_r","joint_knee_r"]
        self.data = []

    def log(self, t, ref_zmp, real_zmp, ref_com, real_com, joint_dict):
        row = {
            "time": t,
            "ref_zmp_x": ref_zmp[0],
            "ref_zmp_y": ref_zmp[1],
            "real_zmp_x": real_zmp[0] if real_zmp is not None else np.nan,
            "real_zmp_y": real_zmp[1] if real_zmp is not None else np.nan,
            "ref_com_x": ref_com[0],
            "ref_com_y": ref_com[1],
            "real_com_x": real_com[0],
            "real_com_y": real_com[1],
            "joint_hip_l": joint_dict.get("left_hip_pitch", np.nan),
            "joint_knee_l": joint_dict.get("left_knee", np.nan),
            "joint_hip_r": joint_dict.get("right_hip_pitch", np.nan),
            "joint_knee_r": joint_dict.get("right_knee", np.nan),
        }
        self.data.append(row)

    def save_csv(self):
        with open(self.save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.data)
        print(f"日志保存至 {self.save_path}")

    def plot(self):
        arr = np.genfromtxt(self.save_path, delimiter=",", skip_header=1)
        t = arr[:,0]

        fig, axes = plt.subplots(3,1, figsize=(10,12))

        # ZMP
        axes[0].plot(t, arr[:,1], label="ref_zmp_x", linestyle="--")
        axes[0].plot(t, arr[:,3], label="real_zmp_x")
        axes[0].set_title("ZMP X")
        axes[0].legend()
        axes[0].grid(True)

        # CoM
        axes[1].plot(t, arr[:,5], label="ref_com_x", linestyle="--")
        axes[1].plot(t, arr[:,7], label="real_com_x")
        axes[1].set_title("CoM X")
        axes[1].legend()
        axes[1].grid(True)

        # 关节
        axes[2].plot(t, arr[:,9], label="left_hip_pitch")
        axes[2].plot(t, arr[:,10], label="left_knee")
        axes[2].set_title("Joint Angles")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("log_plot.png", dpi=150)
        plt.show()
