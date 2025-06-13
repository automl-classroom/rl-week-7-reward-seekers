import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def load_training_data(
    data_pattern="rl_exercises/week_7/data/training_data_seed_*.csv",
    rnd_pattern="rl_exercises/week_7/data/rnd_losses_seed_*.csv",
):
    training_files = glob.glob(data_pattern)
    rnd_files = glob.glob(rnd_pattern)

    all_dfs = []
    for i, file in enumerate(sorted(training_files)):
        # print(f'training from {file} {i+1})')
        df = pd.read_csv(file)
        seed = int(file.split("_")[-1].replace(".csv", ""))
        df["seed"] = seed
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    return combined_df, rnd_files


def plot_training_curves(df):
    plt.figure(figsize=(12, 4))

    # External rewards
    plt.subplot(1, 2, 1)
    plt.plot(df["steps"], df["rewards"], alpha=0.8)
    plt.xlabel("Steps")
    plt.ylabel("External Reward")
    plt.title("External Rewards")
    plt.grid(True, alpha=0.3)

    # Intrinsic rewards
    plt.subplot(1, 2, 2)
    plt.plot(df["steps"], df["intrinsic_rewards"], alpha=0.8)
    plt.xlabel("Steps")
    plt.ylabel("Intrinsic Reward")
    plt.title("Intrinsic Rewards")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_rnd_losses(rnd_files):
    plt.figure(figsize=(10, 6))

    for file in sorted(rnd_files):
        df = pd.read_csv(file)
        seed = int(file.split("_")[-1].replace(".csv", ""))
        plt.plot(df["rnd_loss"], alpha=0.7, label=f"Seed {seed}")

    plt.xlabel("RND Update Steps")
    plt.ylabel("RND Loss (MSE)")
    plt.title("RND Prediction Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.show()


def create_rliable_plots(steps, scores_dict):
    iqm = lambda scores: np.array(
        [
            metrics.aggregate_iqm(scores[:, eval_idx])
            for eval_idx in range(scores.shape[-1])
        ]
    )

    iqm_scores, iqm_cis = get_interval_estimates(
        scores_dict,
        iqm,
        reps=2000,
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    plot_sample_efficiency_curve(
        steps,
        iqm_scores,
        iqm_cis,
        algorithms=list(scores_dict.keys()),
        xlabel="Environment Steps",
        ylabel="IQM Episode Reward",
        title="RND-DQN Sample Efficiency",
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    result = load_training_data()

    df, rnd_files = result

    print("Plot training")
    plot_training_curves(df)

    print("Plotting RND losse")
    plot_rnd_losses(rnd_files)


if __name__ == "__main__":
    main()
