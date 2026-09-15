# 🤖 Robot Projects / 机器人项目资料库

> 一个面向机器人研发、机械设计、三维建模与机器人学习的开源资料仓库。
> 本仓库持续整理和保存机械臂、人形机器人、轮式机器人及相关三维模型、工程文件和设计资料。

[![GitHub](https://img.shields.io/badge/GitHub-Lijiahao--robot-blue?logo=github)](https://github.com/Lijiahao-robot)
[![Repository](https://img.shields.io/badge/Repository-robort--123456-green)](https://github.com/Lijiahao-robot/robort-123456)

---

## 📖 项目简介

`robort-123456` 是一个机器人技术资料与工程文件集合，主要用于：

* 🤖 人形机器人研究与设计
* 🦾 机械臂结构设计
* 🚗 轮式机器人相关资料整理
* 🧩 机器人三维建模与装配
* 🖨️ 3D 打印机器人结构设计
* 🔧 CNC 加工零件设计
* 📐 SolidWorks / STEP 三维模型管理
* 💻 机器人软硬件开发资料整理

仓库中的内容以实际工程文件和机器人设计资料为主，适合机器人爱好者、工程技术人员和学生进行学习、研究与二次开发。

---

## 📂 仓库结构

```text
robort-123456/
│
├── ARM-main/
│   └── 3D/
│       ├── sw/             # SolidWorks 三维模型
│       └── 加工/
│           ├── cnc/        # CNC 加工文件
│           └── 打印件/     # 3D 打印零件
│
├── 人形机器人22/
│   └──                    # 人形机器人结构、模块及设计资料
│
├── 开源3D打印人形机器人Berkeley Humanoid Lite三维图纸 STEP格式 附源码/
│   └──                    # Berkeley Humanoid Lite 相关模型及源码
│
├── 轮式机器人/
│   └──                    # 轮式机器人相关资料
│
├── LICENSE
└── README.md
```

其中 `ARM-main/3D` 目录目前包含多个 SolidWorks 装配体和零件模型，以及用于 CNC 加工的 STEP 文件，例如：

```text
ARM-main/
└── 3D/
    ├── sw/
    │   ├── PYF42-L2.SLDASM
    │   ├── ZDT_X42_L40Motor.SLDASM
    │   ├── Waist.SLDPRT
    │   ├── j1_1.SLDPRT
    │   ├── j2.SLDPRT
    │   ├── j3_1.SLDPRT
    │   ├── j4.SLDPRT
    │   └── ...
    │
    └── 加工/
        └── cnc/
            ├── Waist.STEP
            ├── j1_1.STEP
            ├── j1_2.STEP
            ├── j2.STEP
            ├── j3_1.STEP
            └── ...
```

---

## 🦾 机械臂

`ARM-main` 主要用于保存机器人机械臂相关的机械设计和三维模型。

目前包含：

* SolidWorks `.SLDPRT` 零件文件
* SolidWorks `.SLDASM` 装配体文件
* STEP `.STEP` / `.step` 工程交换格式
* 电机及减速机构设计
* 关节结构设计
* 腰部及基座结构
* CNC 加工零件

这些文件可以用于：

1. 机器人机械结构研究
2. CAD 二次设计
3. CNC 加工准备
4. 3D 打印制造
5. 机器人原型开发

---

## 🤖 人形机器人

`人形机器人22/` 目录保存了人形机器人相关的设计资料和结构图片。

包括但不限于：

* 机器人整体结构
* 头部模块
* 执行器与稳定机构
* 电池与电源模块
* 传感器布局
* 激光雷达相关设计
* 自动充电结构
* 显示器与电池状态信息
* 机器人结构示意图

例如：

```text
人形机器人22/
├── Front view.jpg
├── Head module.jpg
├── Head module exploded.jpg
├── Actuators for stabilization.jpg
├── Automatic docking for charging and battery compartment for ,lidar sensors.jpg
└── ...
```

---

## 🖨️ Berkeley Humanoid Lite

仓库中还包含：

> **开源 3D 打印人形机器人 Berkeley Humanoid Lite**

相关目录用于保存三维图纸、STEP 文件以及配套源码/资料。

该部分适合用于：

* 人形机器人结构学习
* 3D 打印制造
* 机器人机械设计研究
* 人形机器人原型开发
* 开源机器人项目学习

---

## 🚗 轮式机器人

`轮式机器人/` 目录用于整理轮式机器人相关工程和资料。

后续可以在此基础上继续扩展：

```text
轮式机器人/
├── CAD/
├── URDF/
├── MATLAB/
├── ROS/
├── Firmware/
├── Simulation/
└── Documentation/
```

---

## 🛠️ 推荐软件

根据不同文件类型，可以使用以下软件打开或处理项目：

| 文件类型             | 推荐软件                              |
| ---------------- | --------------------------------- |
| `.SLDPRT`        | SolidWorks                        |
| `.SLDASM`        | SolidWorks                        |
| `.STEP` / `.STP` | SolidWorks / FreeCAD / Fusion 360 |
| `.STL`           | MeshLab / Blender / FreeCAD       |
| `.URDF`          | ROS / Gazebo / RViz               |
| `.MAT` / `.M`    | MATLAB                            |
| `.PY`            | Python                            |

---

## 🚀 使用方法

### 1. 克隆仓库

```bash
git clone https://github.com/Lijiahao-robot/robort-123456.git
cd robort-123456
```

### 2. 浏览项目

根据需要进入对应目录：

```bash
cd ARM-main
```

或者：

```bash
cd 人形机器人22
```

### 3. 打开三维模型

SolidWorks 用户可以直接打开：

```text
.SLDPRT
.SLDASM
```

其他 CAD 软件可以优先使用：

```text
.STEP
.STP
```

进行模型导入。

---

## 🔬 项目方向

本仓库后续将继续围绕以下方向完善：

* [ ] 人形机器人整机设计
* [ ] 六自由度机械臂
* [ ] 人形机器人腿部机构
* [ ] 机器人关节模组
* [ ] 电机与减速器设计
* [ ] ROS / ROS 2
* [ ] URDF / Xacro
* [ ] Gazebo 仿真
* [ ] MATLAB / Simulink
* [ ] MPC 运动控制
* [ ] 机器人视觉
* [ ] SLAM / Navigation
* [ ] 机器人运动规划
* [ ] 机器人动力学与控制
* [ ] 3D 打印机器人
* [ ] CNC 精密加工

---

## 📌 注意事项

本仓库中的不同目录可能来自不同阶段的设计、实验和资料整理。

部分模型和资料可能仍处于开发阶段，因此：

* 使用前请确认对应文件的版本；
* 机械加工前请再次检查尺寸、公差和材料；
* 实际装配时请根据具体硬件进行验证；
* 不同 CAD 软件打开同一模型时可能存在兼容性差异。

---

## 📄 License

本仓库包含一个 `LICENSE` 文件。

使用、修改或再次发布仓库中的代码、模型和工程文件时，请根据对应目录及文件的许可证要求进行使用。

---

## 👤 作者

**Lijiahao-robot**

GitHub：

https://github.com/Lijiahao-robot

项目地址：

https://github.com/Lijiahao-robot/robort-123456

---

## ⭐ Star

如果这个仓库对你的机器人学习、机械设计或项目开发有帮助，欢迎给项目点一个 ⭐ **Star**。

欢迎持续关注本项目后续更新。

