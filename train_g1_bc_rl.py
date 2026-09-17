import sys
sys.path.append(".")

from motion.g1_generate_expert_dataset import generate_g1_expert_dataset
from rl.bc_rl import PPOAgent, bc_pretrain
from main_g1_dcm_preview import HumanoidG1Env
import torch

def main():
    device = "cpu"
    # 1.生成G1专家数据集
    generate_g1_expert_dataset(save_path="g1_expert_dataset.pkl", n_cycles=8)

    # 2.初始化G1环境
    env = HumanoidG1Env("./models/g1.xml")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # 3.BC预训练，使用G1专家数据集
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    bc_pretrain(agent.model, expert_path="g1_expert_dataset.pkl", epochs=120, lr=1e-3, device=device)
    agent.save("g1_bc_pretrain.pt")

    # 4.PPO强化学习微调
    total_episodes = 200
    rollout_steps = 2048

    for epi in range(total_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        ep_reward = 0.0

        while not done and step_cnt < rollout_steps:
            action = agent.model.get_action(obs)
            noise = torch.randn_like(torch.tensor(action)) * 0.12
            action_noisy = action + noise.numpy()

            obs_next, reward, done, real_zmp, ref_zmp, real_com = env.step(action_noisy)
            ep_reward += reward

            logprob = -0.5 * ((action_noisy - action)**2).sum()
            value = agent.model.critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()

            agent.collect_step(obs, action_noisy, logprob, reward, done, value)
            obs = obs_next
            step_cnt += 1

        agent.update()
        print(f"Episode {epi:3d} | Reward: {ep_reward:.2f}")
        if (epi+1) % 20 == 0:
            agent.save("g1_bc_rl_final.pt")

    agent.save("g1_bc_rl_final.pt")
    env.logger.save_csv()
    env.logger.plot()

if __name__ == "__main__":
    main()
