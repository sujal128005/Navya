# Project Navyam: Multi-Agent Autonomous System

**Date:** January 2, 2026
**Lead:** Sujal (System Architect)

## 1. Executive Summary
Project Navyam aims to design and implement a cooperative multi-agent autonomous system consisting of one UAV (Aerial Search) and one UGV (Ground Navigation). To ensure 100% accuracy and research-grade reliability, this project strictly follows the V-Model of Systems Engineering.

**Strategy:** "Digital Twin First." Every sensor, motor, and line of code is validated in a high-fidelity simulation (Gazebo Fortress) before touching physical hardware.

## 2. Technology Stack
| Layer | Tool / Software | Function |
| :--- | :--- | :--- |
| **Middleware** | ROS 2 (Humble/Jazzy) | Inter-agent communication |
| **UAV Flight Stack** | PX4 Autopilot | Offboard control logic |
| **Simulation** | Gazebo Fortress | Physics & World rendering |
| **UGV Firmware** | PlatformIO + Arduino | Embedded motor control |
| **Perception** | LIDAR (RPLIDAR) | SLAM & Obstacle Detection |
| **Visualization** | RViz2 & Foxglove | Sensor data debugging |
| **Code Quality** | SonarLint / Flakes | Automated linting |

## 3. Team Structure
* **M1 Sujal (Lead):** System Architect, TF Tree, Global State Machine.
* **M2 (Pilot):** UAV Control, PX4, Flight Safety.
* **M3 (Vision):** Computer Vision, Object Detection, Coordinate Transformation.
* **M4 (Driver):** UGV Navigation, SLAM, Path Planning.
* **M5 (Sim Lead):** Gazebo World Design, Stress Testing.
* **M6 (Hardware):** Embedded Firmware, URDF Modeling, Assembly.
