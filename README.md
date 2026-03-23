# Distributed MPC for Coordinated Path-Following of UAV Systems

> Simulation code accompanying the paper **"Distributed MPC For Coordinated Path-Following"** by Lusine Poghosyan, Anna Manucharyan, Mikayel Aramyan, Naira Hovakimyan, and Tigran Bakaryan.

---

## Overview

This repository provides a high-fidelity simulation framework for **distributed model predictive control (DMPC)** applied to multi-UAV time coordination. Built on top of [RotorPy](https://github.com/spencerfolk/rotorpy), an open-source multirotor simulator, the code replicates the full pipeline described in the paper: from virtual-time coordination to game-theoretic corridor navigation.


## Repository Structure

```
.
├── rotorpy/
│   ├── examples/
│   │   └── coordination_with_mpc.py   # Main entry point
│   └── rotorpy/
│       ├── mpc.py                     # Core DMPC logic
│       └── config.py                  # Simulation configuration
```

### Branches

Each family of simulation scenarios is maintained in a dedicated branch for clarity and reproducibility:

| Branch | Description |
|--------|-------------|
| `main` | Baseline synchronization and MPC horizon sensitivity analysis |
| `sequential` | Corridor navigation with **offline predefined ordering** (sequential subgame) and **autonomous game-theoretic ordering** |

> The corridor navigation scenarios — both the offline ordering and the competitive game formulation — can be found in the **`sequential` branch**.

---

## Simulation Scenarios

### 1. Corridor Navigation — Offline Ordering

The first approach embeds a **predefined UAV ordering** directly into the cost function. Each UAV is penalized for deviating from a prescribed separation $\Delta$ relative to its neighbors during the corridor phase.


---

https://github.com/user-attachments/assets/0b699561-06b5-4ffa-bad4-76d8581f6b1f



### 2. Corridor Navigation — Autonomous Game-Theoretic Ordering

The second approach replaces the offline ordering with a **competitive game formulation**. Inside the corridor, each UAV is penalized for every neighbor ahead of it, creating a natural racing incentive. Outside the corridor, the standard coordination term takes over, driving the swarm back to synchronization.


https://github.com/user-attachments/assets/855db060-7565-4ac3-a1e7-ea5b92709eec


---

## Getting Started

### Prerequisites

- Python 3.8+
- [RotorPy](https://github.com/spencerfolk/rotorpy) installed and configured
- Standard scientific Python stack: `numpy`, `scipy`, `matplotlib`

### Installation

```bash
git clone https://github.com/amanucha/rotorpy_coordination.git
cd rotorpy_coordination
pip install -r requirements.txt
```

### Running the Main Simulation

```bash
python rotorpy/examples/coordination_with_mpc.py
```

### Running the Corridor Navigation Scenarios

```bash
git checkout sequential
python rotorpy/examples/coordination_with_mpc.py
```

Configuration parameters (number of agents, prediction horizon, cost function weights, $\gamma_1^*$, $\gamma_2^*$, $\Delta$, $\bm{\psi}$) can be adjusted in `rotorpy/rotorpy/config.py`.

---
