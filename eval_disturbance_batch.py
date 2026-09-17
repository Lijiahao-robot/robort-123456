import numpy as np
import csv
import os
from main_g1_dcm_preview import HumanoidG1Env
import torch

def run_single_test(env, disturb_force, test_duration=8.0):
    obs = env.reset()
    env.disturb.reset()
    t0 = env.t_sim
    env.disturb.apply_impulse_disturbance(t0+1.5, impulse_force=np.array([disturb_force,0,0]), duration=0.2)

    is_fall = False
    zmp_margin_list = []
    recovery_success = False

    while env.t_sim < t0 + test_duration:
        action = env.agent.get_action(obs)
        obs_next, reward, done, real_zmp, ref_zmp, real_com = env.step(action)
        if real_zmp is not None:
            zmp_margin = env.zmp_metric.compute_margin(real_zmp, env.support_sm.get_support_polygon())[0]
            zmp_margin_list.append(zmp_margin)
        if done:
            is_fall = True
            break
        obs = obs_next

    if not is_fall and env.gait_recovery.recovery_flag is False:
        recovery_success = True

    avg_margin = np.mean(zmp_margin_list) if len(zmp_margin_list) > 0 else -1.0
    return {
        "disturb_force": disturb_force,
        "is_fall": is_fall,
        "avg_zmp_margin": avg_margin,
        "recovery_success": recovery_success
    }

def batch_eval():
    env = HumanoidG1Env("./models/g1.xml")
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    agent = torch.load("g1_bc_rl_final.pt", map_location="cpu")
    env.agent = agent

    force_list = [20,40,60,80,100]
    results = []
    for f in force_list:
        res = run_single_test(env, disturb_force=f, test_duration=8.0)
        results.append(res)
        print(f"扰动力{f}N | 摔倒:{res['is_fall']} | 平均ZMP裕度:{res['avg_zmp_margin']:.3f} | 恢复成功:{res['recovery_success']}")

    # 保存csv
    with open("batch_eval_result.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # 输出markdown报告
    report = "# G1扰动抗干扰批量测试报告\n\n"
    report += "|扰动力(N)|是否摔倒|平均ZMP裕度|恢复成功|\n"
    report += "|---|---|---|---|\n"
    for r in results:
        report += f"|{r['disturb_force']}|{r['is_fall']}|{r['avg_zmp_margin']:.3f}|{r['recovery_success']}|\n"

    total = len(results)
    fall_cnt = sum([1 for r in results if r["is_fall"]])
    recover_cnt = sum([1 for r in results if r["recovery_success"]])
    report += f"\n- 总测试次数：{total}\n"
    report += f"- 摔倒次数：{fall_cnt}，摔倒率：{fall_cnt/total:.2%}\n"
    report += f"- 恢复成功次数：{recover_cnt}\n"

    with open("disturbance_report.md","w",encoding="utf‑8") as f:
        f.write(report)
    print("\n✅批量评估完成，输出：batch_eval_result.csv、disturbance_report.md")

if __name__ == "__main__":
    batch_eval()
