import sys
sys.path.append(".")

from motion.generate_expert_dataset import generate_expert_dataset
from rl.bc_rl import PPOAgent, bc_pretrain
from main_dcm_lipm_ik_wbc_rl_log import HumanoidDCMEnv
import torch

def main():
    device = "cpu"
    # 1. 生成LIPM‑DCM专家数据集
    generate_expert_dataset(save_path="expert_dataset.pkl", n_cycles=10)

    # 2. 创建环境，获取obs/act维度
    env = HumanoidDCMEnv("./models/humanoid.xml")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # 3. PPO Agent + BC预训练
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    bc_pretrain(agent.model, expert_path="expert_dataset.pkl", epochs=100, lr=1e-3, device=device)
    agent.save("bc_pretrain.pt")

    # 4. PPO强化学习微调
    total_episodes = 200
    rollout_steps = 2048

    for epi in range(total_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        ep_reward = 0.0

        while not done and step_cnt < rollout_steps:
            action = agent.model.get_action(obs)
            # 简单高斯采样
            noise = torch.randn_like(torch.tensor(action)) * 0.1
            action_noisy = action + noise.numpy()

            obs_next, reward, done, real_zmp, ref_zmp, real_com, ref_com = env.step(action_noisy)
            ep_reward += reward

            # 计算logprob，简化
            logprob = -0.5 * ((action_noisy - action)**2).sum()
            value = agent.model.critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()

            agent.collect_step(obs, action_noisy, logprob, reward, done, value)
            obs = obs_next
            step_cnt += 1

        # 每轮结束更新PPO
        agent.update()
        print(f"Episode {epi:3d} | Reward: {ep_reward:.2f}")

        if (epi+1) % 20 == 0:
            agent.save("bc_rl_final.pt")

    agent.save("bc_rl_final.pt")
    env.logger.save_csv()
    env.logger.plot()

if __name__ == "__main__":
    main()
