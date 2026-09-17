import torch
import numpy as np
from rl.bc_rl import BCAgent
from utils.domain_randomizer import DomainRandomizer

def train_sim2real(env, epochs=100, batch_size=256):
    # 1. 域随机化开启
    env.domain_rand.enable = True
    agent = BCAgent(obs_dim=env.obs_dim, act_dim=env.act_dim)

    # 阶段1：BC预训练（域随机化仿真）
    print("===== Phase1: BC Pre‑training on randomized sim =====")
    for epoch in range(epochs):
        # 收集随机化环境样本
        traj_list = []
        for _ in range(20):
            obs = env.reset()
            done=False
            while not done:
                action = agent.get_action(obs)
                obs_next, r, done, _, _, _ = env.step(action)
                traj_list.append((obs,action))
                obs=obs_next
        agent.train_step(traj_list, batch_size=batch_size)
        print(f"Epoch {epoch} BC loss: {agent.get_loss():.4f}")

    # 阶段2：域自适应迁移学习（简单域对抗）
    print("===== Phase2: Sim‑to‑Real domain adaptation =====")
    # 此处可以接入域对抗DANN，缩小仿真与真实的特征分布
    # 保存迁移模型
    torch.save(agent.state_dict(), "g1_sim2real_model.pt")
    print("✅ Sim‑to‑Real model saved: g1_sim2real_model.pt")
    return agent

if __name__ == "__main__":
    pass

