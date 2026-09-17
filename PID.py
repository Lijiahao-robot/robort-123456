import numpy as np

class PID:
    def __init__(self, kp, ki, kd, max_output=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, setpoint, measurement, dt):
        """
        setpoint:目标值
        measurement:当前测量值
        dt:时间步长
        return:控制输出
        """
        error = setpoint - measurement
        # 积分项
        self.integral += error * dt
        # 微分项
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 限幅
        output = np.clip(output, -self.max_output, self.max_output)

        self.prev_error = error
        return output


# ========== 使用示例：姿态控制（机器人俯仰姿态） ==========
if __name__ == "__main__":
    pid = PID(kp=2.0, ki=0.1, kd=0.5, max_output=5.0)
    target_angle = 0.0
    current_angle = 0.8
    dt = 0.01

    for _ in range(200):
        u = pid.step(target_angle, current_angle, dt)
        current_angle += u * dt
        print(f"angle:{current_angle:.3f}, ctrl:{u:.3f}")
