# DDPG Actor-Critic for DeepMind Control Suite Cartpole Balance

This repository contains a full Deep Deterministic Policy Gradient (DDPG) actor–critic implementation used to train a continuous-control policy for the DeepMind Control Suite **cartpole/balance** task.  
It includes training on seeds **0, 1, 2**, evaluation on seed **10**, mean ± std learning curves.

## Description
- Implements DDPG (actor, critic, target networks, replay buffer, soft updates)

- Trains using three seeds (0, 1, 2) for statistical reproducibility

- Evaluates using one seed (10), as required by assignment

- Saves all training/evaluation logs in ddpg_results/

- Hyperparameter tables and plot figures

## 1. Environment Setup
```bash
pip install torch numpy matplotlib dm-control
```

## 2. How to Run

```bash
python ddpg_cartpole_dmcontrol.py
python plot_results.py
```

## 3. Output Plots
- ddpg_results/train_learnining_curve.png - Training curve (mean ± std)
- ddpg_results/eval_returns_seeds10.png - Evaluation returns

## 4. References
- Lillicrap et al., “Continuous Control with Deep Reinforcement Learning,” 2016.
- Tassa et al., “DeepMind Control Suite,” 2018.


