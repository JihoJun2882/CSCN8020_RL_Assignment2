# Q-Learning for Taxi-v3 Environment
## Course: CSCN8020 - Reinforcement Learning

This repository contains an implementation and sensitivity analysis of the **Q-Learning algorithm** applied to the `Taxi-v3` environment from OpenAI Gym (Gymnasium). The project focuses on optimizing hyperparameters to achieve efficient passenger delivery with minimal steps.

---

## 1. Project Overview
The objective is to train a reinforcement learning agent to navigate a 5x5 grid, pick up a passenger from a designated location, and drop them off at a target destination.

### Environment Specifications:
- **State Space (500):** Discrete states encoding taxi position, passenger location, and destination.
- **Action Space (6):** Move South(0), North(1), East(2), West(3), Pickup(4), Drop-off(5).
- **Rewards:** - `-1` for every step.
  - `+20` for successful delivery.
  - `-10` for illegal pickup or drop-off actions.

---

## 2. Methodology: Q-Learning
The agent uses a **Q-Table** to learn the optimal policy $\pi^*$. The Q-values are updated using the Temporal Difference (TD) formula:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### Hyperparameters:
- **Learning Rate ($\alpha$):** Determines how much new information overrides old information.
- **Exploration Factor ($\epsilon$):** Controls the balance between exploring new actions and exploiting known rewards (Epsilon-greedy policy).
- **Discount Factor ($\gamma$):** Fixed at **0.9** to prioritize long-term rewards.

---

## 3. Sensitivity Analysis
We performed a systematic study by varying one parameter at a time to observe its impact on convergence:

| Parameter | Tested Values | Finding |
|-----------|---------------|---------|
| **Learning Rate ($\alpha$)** | 0.01, 0.001, 0.1, 0.2 | $\alpha=0.2$ showed the fastest convergence and highest reward. |
| **Exploration Factor ($\epsilon$)** | 0.1, 0.2, 0.3 | $\epsilon=0.1$ was optimal; higher values led to unnecessary random detours. |

---

## 4. Key Results
- **Optimal Configuration:** $\alpha=0.2, \epsilon=0.1, \gamma=0.9$.
- **Performance:** The optimized agent achieved an average return of approximately **+2.57** (baseline) to significantly higher efficiency after tuning, reducing average steps from over 30 to **~23.6**.
- **Conclusion:** Higher learning rates significantly accelerate convergence in this deterministic discrete environment, while low exploration prevents the agent from deviating from the optimal path once discovered.

---

## 5. Getting Started
### Prerequisites
- Python 3.8+
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib

### Installation
```bash
pip install gymnasium numpy matplotlib
```

### Running the Notebook
Open `Assignmnet2.ipynb` in your Jupyter environment and run the cells sequentially to observe:
1. Baseline training.
2. Sensitivity analysis plots.
3. Final agent simulation.

---

## 6. Visualizations
The project includes plots for:
- **Average Return per Episode:** Tracking learning progress.
- **Total Steps per Episode:** Measuring navigation efficiency.
