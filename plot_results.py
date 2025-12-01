import numpy as np
import matplotlib.pyplot as plt

data_dir = "ddpg_results"

mean_train = np.load(f"{data_dir}/train_mean.npy")
std_train = np.load(f"{data_dir}/train_std.npy")

x = np.arange(len(mean_train))

plt.figure()
plt.plot(x, mean_train, label="Train mean return")
plt.fill_between(x, mean_train - std_train, mean_train + std_train, alpha=0.3)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DDPG Cartpole-Balance: Training Performance (mean ± std)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{data_dir}/train_learning_curve.png", dpi=300)
plt.close()

# Simple eval plot from eval_returns_seed10.npy
eval_returns = np.load(f"{data_dir}/eval_returns_seed10.npy")
episodes_eval = np.arange(len(eval_returns))

plt.figure()
plt.plot(episodes_eval, eval_returns, marker="o", label="Eval returns (seed 10)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DDPG Cartpole-Balance: Evaluation (seed 10)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{data_dir}/eval_returns_seed10.png", dpi=300)
plt.close()
