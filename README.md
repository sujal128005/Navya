# Project Navyam: Cooperative Multi-Agent Autonomous System

![Status](https://img.shields.io/badge/Status-Phase%201%3A%20Setup-blue)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble-green)
![Simulation](https://img.shields.io/badge/Simulation-Gazebo%20Fortress-orange)

## 1. Executive Summary
**Project Navyam** is a research initiative to design and implement a cooperative autonomous system consisting of one **UAV (Aerial Search)** and one **UGV (Ground Navigation)**. 

To ensure 100% accuracy and reliability, this project follows the **V-Model of Systems Engineering**. We prioritize a **"Digital Twin First" strategy**: every sensor, motor, and line of code is validated in a high-fidelity simulation (Gazebo Fortress) before deployment on physical hardware.

### System Objectives
1.  **Search:** UAV performs an aerial sweep to detect a target (Red Box) using Computer Vision.
2.  **Locate:** UAV converts pixel data into global geospatial coordinates (GPS/Cartesian).
3.  **Navigate:** UGV receives coordinates and autonomously navigates to the target using LiDAR-based SLAM and path planning.
4.  **Cooperate:** Agents share map data to ensure safe operations.

---

## 2. Technology Stack & Environment
All team members must adhere to this standardized environment to prevent compatibility issues.

| Layer | Tool / Software | Description |
| :--- | :--- | :--- |
| **OS** | Ubuntu 22.04 LTS | Standard Operating System (Jammy Jellyfish) |
| **Middleware** | ROS 2 (Humble) | Inter-agent communication backbone |
| **UAV Stack** | PX4 Autopilot | Offboard flight control & failsafes |
| **Simulation** | Gazebo Fortress | Physics engine & World rendering |
| **UGV Firmware** | PlatformIO + Arduino | Low-level motor control & Odometry |
| **Perception** | OpenCV / YOLO | Object detection & State estimation |
| **Navigation** | Nav2 Stack + SLAM Toolbox | Path planning & Mapping |
| **Visualization** | RViz2 | Real-time sensor data debugging |

---

## 3. Directory Structure
The team must strictly follow this folder hierarchy to prevent merge conflicts.

```text
Navyam_Project/
├── navyam_ws/                 # The ROS 2 Workspace (The Brain)
│   ├── src/
│   │   ├── navyam_interfaces/ # Custom Msg/Srv definitions
│   │   ├── navyam_simulation/ # Worlds, Launch files, Models
│   │   ├── navyam_uav_control/# Drone logic (Python/C++)
│   │   └── navyam_ugv_control/# Car logic (Nav2 integration)
├── hardware_specs/            # Architecture diagrams & datasheets
├── embedded_firmware/         # Arduino Code (PlatformIO)
└── docs/                      # Phase documentation
