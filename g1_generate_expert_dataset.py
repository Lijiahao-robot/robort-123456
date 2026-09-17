import numpy as np
import pickle
import mujoco
from motion.foot_trajectory import FootTrajectoryGenerator
from motion.dcm_kajita_preview import DCMKajitaPreview
from motion.ik_g1 import HumanoidIK_G1

def generate_g1_expert_dataset(model_path="./models/g1.xml", save_path="g1_expert_dataset.pkl", n_cycles=8):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    foot_traj = FootTrajectoryGenerator(step_height=0.09, dt=dt)
    dcm_preview = DCMKajitaPreview(com_height=0.92, preview_N=15, dt=dt)
    ik = HumanoidIK_G1()

    obs_list = []
    act_list = []

    mujoco.mj_resetData(model, data)
    t_sim = 0.0
    cycle_duration = 1.4  # 单步步态周期

    for cycle in range(n_cycles):
        t_start_cycle = t_sim
        # 摆动脚起点、终点
        foot_start = np.array([0.0, 0.08, 0.0])
        foot_end = np.array([0.28, 0.08, 0.0])

        for _ in range(int(cycle_duration/dt)):
            # 足部轨迹
            foot_pos, _ = foot_traj.generate_foot_trajectory(
                t_sim,
                t_swing_start=t_start_cycle,
                t_swing_end=t_start_cycle+0.7,
                foot_start=foot_start,
                foot_end=foot_end
            )
            # 参考ZMP序列
            zmp_ref_seq = np.ones(dcm_preview.N)*0.14

            # 当前CoM
            com = data.subtree_com[model.body("root").id][:3]
            dcom = np.zeros(3)
            xi = dcm_preview.compute_dcm_state(com[0], dcom[0])
            zmp_ctrl = dcm_preview.get_current_zmp(xi, zmp_ref_seq)

            # IK得到G1关节角度
            com_target = np.array([0.0,0.0,0.92])
            l_foot = foot_pos
            r_foot = np.array([0.0, -0.08, 0.0])
            joint_dict = ik.compute_joint_from_com(com_target, l_foot, r_foot)

            # 构造专家动作：G1关节角度向量
            expert_action = np.zeros(model.nu)
            expert_action[0] = joint_dict["left_hip_yaw"]
            expert_action[1] = joint_dict["left_hip_pitch"]
            expert_action[2] = joint_dict["left_knee"]
            expert_action[3] = joint_dict["left_ankle"]
            expert_action[4] = joint_dict["right_hip_yaw"]
            expert_action[5] = joint_dict["right_hip_pitch"]
            expert_action[6] = joint_dict["right_knee"]
            expert_action[7] = joint_dict["right_ankle"]

            obs = np.concatenate([data.qpos, data.qvel])
            obs_list.append(obs)
            act_list.append(expert_action)

            data.ctrl[:] = expert_action
            mujoco.mj_step(model, data)
            t_sim += dt

    dataset = {
        "observations": np.array(obs_list),
        "actions": np.array(act_list)
    }
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"G1专家数据集保存至 {save_path}, 样本数:{len(obs_list)}")

if __name__ == "__main__":
    generate_g1_expert_dataset()
