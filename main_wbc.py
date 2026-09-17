import mujoco
import mujoco.viewer
import numpy as np
from controllers.wbc_qp import WBC_QP

model = mujoco.MjModel.from_xml_path("./models/humanoid.xml")
data = mujoco.MjData(model)
wbc = WBC_QP(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 期望：基座保持水平，关节尽量保持0
        desired_base_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        desired_joint_vel = np.zeros(model.nu)

        tau = wbc.solve(desired_base_vel, desired_joint_vel)
        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        viewer.sync()
