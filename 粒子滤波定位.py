
import numpy as np
import random

class ParticleFilterLocalization:
    def __init__(self, num_particles, init_pose, noise_std):
        self.num_particles = num_particles
        self.particles = np.random.normal(init_pose, noise_std, (num_particles,3))
        self.weights = np.ones(num_particles)/num_particles

    def motion_update(self, u, motion_noise):
        """运动更新：里程计噪声"""
        dx, dy, dtheta = u
        for i in range(self.num_particles):
            self.particles[i,0] += dx + np.random.normal(0, motion_noise[0])
            self.particles[i,1] += dy + np.random.normal(0, motion_noise[1])
            self.particles[i,2] += dtheta + np.random.normal(0, motion_noise[2])

    def measurement_update(self, z, landmark, obs_noise):
        """观测更新，根据路标观测更新权重"""
        for i in range(self.num_particles):
            px, py, _ = self.particles[i]
            dist_pred = np.linalg.norm(np.array([px,py])-np.array(landmark))
            dist_obs = z
            #高斯似然
            prob = np.exp(-0.5 * ((dist_pred-dist_obs)/obs_noise)**2)
            self.weights[i] *= prob
        self.weights /= np.sum(self.weights)

    def resample(self):
        """重采样"""
        new_particles = []
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        for idx in indices:
            new_particles.append(self.particles[idx])
        self.particles = np.array(new_particles)
        self.weights = np.ones(self.num_particles)/self.num_particles

    def get_estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


if __name__ == "__main__":
    pf = ParticleFilterLocalization(num_particles=200, init_pose=[0,0,0], noise_std=[0.1,0.1,0.1])
    landmark = np.array([3, 2])
    pf.motion_update(u=[0.2,0,0], motion_noise=[0.05,0.05,0.02])
    pf.measurement_update(z=3.2, landmark=landmark, obs_noise=0.2)
    pf.resample()
    est = pf.get_estimate()
    print("粒子滤波估计位姿：", est)
