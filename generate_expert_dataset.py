import pickle
import numpy as np
from motion.dcm_lipm_gait import DCMLIPMGaitPlanner
from motion.ik import HumanoidIK

def generate_expert_dataset(save_path="expert_dataset.pkl", n_cycles=10):
    planner = DCMLIPMGaitPlanner(
        com_height=0.8, step_length=0.25, step_width=0.14,
        T_support=0.7, T_double=0.2, dt=0.01
    )
    ik = HumanoidIK(thigh_len=0.4, shank_len=0.4)

    obs_list = []
    act_list = []

    for cycle in range(n_cycles):
        t_list, ref_zmp_x, ref_zmp_y, cm_x, cm_y, _, _ = planner.plan_one_step()
        N = len(t_list)
        for i in range(N):
            com_target = np.array([cm_x[i], cm_y[i], 0.8])
            # 简化脚位置
            left_foot = np.array([cm_x[i]-0.05, 0.07, 0.0])
            right_foot = np.array([cm_x[i]-0.05, -0.07, 0.0])
            joint_dict = ik.compute_joint_from_com(com_target, left_foot, right_foot)
            # 构造观测（简化：com位置+速度）
            obs = np.concatenate([com_target, np.array([0.0,0.0,0.0])])
            action = np.array([
                joint_dict["left_hip_pitch"],
                joint_dict["left_knee"],
                joint_dict["left_ankle"],
                joint_dict["right_hip_pitch"],
                joint_dict["right_knee"],
                joint_dict["right_ankle"],
            ])
            obs_list.append(obs)
            act_list.append(action)

    dataset = {
        "observations": np.array(obs_list),
        "actions": np.array(act_list)
    }
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"专家数据集生成完毕，保存 {save_path}，样本数 {len(obs_list)}")
    return dataset

if __name__ == "__main__":
    generate_expert_dataset(n_cycles=10)
