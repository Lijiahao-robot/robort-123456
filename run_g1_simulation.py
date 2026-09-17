
import argparse
import subprocess
import sys
import os

def check_dependencies():
    required = ["torch","mujoco","numpy","matplotlib","cvxpy","osqp"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if len(missing) > 0:
        print(f"缺少依赖包：{missing}")
        print("执行安装：pip install mujoco torch numpy matplotlib cvxpy osqp")
        sys.exit(1)
    print("✅ 依赖检查通过")

def run_train():
    print("===== 开始G1 BC‑RL训练 =====")
    subprocess.run([sys.executable, "train_g1_bc_rl.py"], check=True)

def run_simulation(mode, step_length, step_freq, disturb_force, test_duration):
    print(f"===== 运行仿真 | 运动模式:{mode} | 步长:{step_length} | 步频:{step_freq} | 扰动力:{disturb_force}N =====")
    # 这里可以把参数传入main_g1_dcm_preview，这里简化为调用主程序
    subprocess.run([sys.executable, "main_g1_dcm_preview.py",
                    "--mode", mode,
                    "--step_length", str(step_length),
                    "--step_freq", str(step_freq),
                    "--disturb_force", str(disturb_force),
                    "--duration", str(test_duration)], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="forward", choices=["forward","lateral","turn_in_place"])
    parser.add_argument("--step_length", default=0.28, type=float)
    parser.add_argument("--step_freq", default=1.2, type=float)
    parser.add_argument("--disturb_force", default=0.0, type=float)
    parser.add_argument("--train", action="store_true", help="是否执行BC‑RL训练")
    parser.add_argument("--test_duration", default=8.0, type=float)
    args = parser.parse_args()

    check_dependencies()
    if args.train:
        run_train()
    run_simulation(args.mode, args.step_length, args.step_freq, args.disturb_force, args.test_duration)

if __name__ == "__main__":
    main()
