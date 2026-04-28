"""Visualize dense reward and subtask labels from a LeRobot parquet dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def plot_episode(df_ep: pd.DataFrame, episode_idx: int, ax_dense, ax_subtask, ax_sparse):
    frames = df_ep["frame_index"].values

    ax_dense.plot(frames, df_ep["dense_rewards"].values, color="#2196F3", linewidth=0.8)
    ax_dense.set_ylabel("Dense Reward", fontsize=9)
    ax_dense.set_title(f"Episode {episode_idx}", fontsize=10, fontweight="bold")
    ax_dense.grid(True, alpha=0.3)

    subtask = df_ep["subtask_idx"].values
    n_subtasks = int(subtask.max()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_subtasks)
    for s in range(n_subtasks):
        mask = subtask == s
        ax_subtask.fill_between(frames, 0, 1, where=mask, color=cmap(s), alpha=0.6, label=f"Subtask {s}")
    ax_subtask.set_ylabel("Subtask", fontsize=9)
    ax_subtask.set_yticks([])
    ax_subtask.legend(loc="upper right", fontsize=6, ncol=min(n_subtasks, 5), framealpha=0.8)
    ax_subtask.grid(True, alpha=0.3)

    ax_sparse.plot(frames, df_ep["sparse_rewards"].values, color="#FF5722", linewidth=0.8)
    ax_sparse.set_ylabel("Sparse Reward", fontsize=9)
    ax_sparse.set_xlabel("Frame Index", fontsize=9)
    ax_sparse.grid(True, alpha=0.3)


def plot_overview(df: pd.DataFrame):
    episodes = sorted(df["episode_index"].unique())
    n_episodes = len(episodes)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Reward Overview Across All Episodes", fontsize=13, fontweight="bold")

    ep_boundaries = []
    dense_all, sparse_all, subtask_all, x_all = [], [], [], []
    offset = 0
    for ep in episodes:
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
        n = len(ep_df)
        x = np.arange(offset, offset + n)
        x_all.append(x)
        dense_all.append(ep_df["dense_rewards"].values)
        sparse_all.append(ep_df["sparse_rewards"].values)
        subtask_all.append(ep_df["subtask_idx"].values)
        ep_boundaries.append(offset)
        offset += n

    x_all = np.concatenate(x_all)
    dense_all = np.concatenate(dense_all)
    sparse_all = np.concatenate(sparse_all)
    subtask_all = np.concatenate(subtask_all)

    n_subtasks = int(subtask_all.max()) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_subtasks)

    for s in range(n_subtasks):
        mask = subtask_all == s
        axes[0].fill_between(x_all, 0, 1, where=mask, color=cmap(s), alpha=0.5, label=f"Subtask {s}")
    axes[0].set_ylabel("Subtask", fontsize=9)
    axes[0].set_yticks([])
    axes[0].legend(loc="upper right", fontsize=7, ncol=min(n_subtasks, 5))
    for b in ep_boundaries:
        axes[0].axvline(b, color="gray", linewidth=0.3, alpha=0.5)

    axes[1].plot(x_all, dense_all, color="#2196F3", linewidth=0.3, alpha=0.7, label="Dense")
    axes[1].plot(x_all, sparse_all, color="#FF5722", linewidth=0.3, alpha=0.7, label="Sparse")
    axes[1].set_ylabel("Reward", fontsize=9)
    axes[1].set_xlabel("Global Frame Index", fontsize=9)
    axes[1].legend(fontsize=8)
    for b in ep_boundaries:
        axes[1].axvline(b, color="gray", linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=str, help="Path to the parquet file")
    parser.add_argument(
        "--episodes", "-e", type=int, nargs="*", default=None,
        help="Episode indices to plot (default: first 4 + overview)",
    )
    parser.add_argument("--save", "-s", type=str, default=None, help="Save figure to path instead of showing")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows, {df['episode_index'].nunique()} episodes")
    print(f"  dense_rewards  range: [{df['dense_rewards'].min():.4f}, {df['dense_rewards'].max():.4f}]")
    print(f"  sparse_rewards range: [{df['sparse_rewards'].min():.4f}, {df['sparse_rewards'].max():.4f}]")
    print(f"  subtask_idx    range: [{df['subtask_idx'].min()}, {df['subtask_idx'].max()}]")

    all_episodes = sorted(df["episode_index"].unique())
    episodes = args.episodes if args.episodes is not None else all_episodes[:4]

    fig_overview = plot_overview(df)

    for ep in episodes:
        if ep not in all_episodes:
            print(f"Warning: episode {ep} not found, skipping")
            continue

        df_ep = df[df["episode_index"] == ep].sort_values("frame_index")
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.suptitle("Reward & Subtask Visualization", fontsize=13, fontweight="bold")
        plot_episode(df_ep, ep, axes[0], axes[1], axes[2])
        fig.tight_layout()

    if args.save:
        fig_overview.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
