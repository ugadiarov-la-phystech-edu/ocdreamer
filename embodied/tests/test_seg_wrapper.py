"""Quick smoke-test for SegmentationMaskWrapper on HomeGrid."""

import os
import gym
import homegrid
import homegrid.wrappers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from embodied.envs.homegrid import SegmentationMaskWrapper

OUT_DIR = "seg_test_images"
os.makedirs(OUT_DIR, exist_ok=True)


def save_seg_image(obs, seg_labels, filename, title=""):
    """Save side-by-side RGB image and colorized segmentation mask."""
    seg_mask = obs["seg_mask"]
    rgb = obs["image"]

    unique_ids = np.unique(seg_mask)
    n_ids = max(unique_ids.max() + 1, 2)
    # Build a distinct colormap: 0=black (background), rest get distinct colors
    base_colors = ["black"] + list(mcolors.TABLEAU_COLORS.values())
    while len(base_colors) < n_ids:
        base_colors.append(plt.cm.Set3(len(base_colors) % 12))
    cmap = mcolors.ListedColormap(base_colors[:n_ids])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    im = axes[1].imshow(seg_mask, cmap=cmap, vmin=0, vmax=n_ids - 1,
                        interpolation="nearest")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    # Legend
    handles = []
    for uid in unique_ids:
        color = cmap(uid / (n_ids - 1)) if n_ids > 1 else cmap(0)
        label = seg_labels.get(uid, f"id={uid}")
        handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                  markerfacecolor=color, markersize=10,
                                  label=f"{uid}: {label}"))
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                   fontsize=7, frameon=False)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# --- create raw homegrid env ---
env = gym.make(
    "homegrid-task",
    disable_env_checker=True,
    max_steps=100,
    num_trashobjs=2,
    num_trashcans=2,
)
env = homegrid.wrappers.Gym26Wrapper(env)

# --- wrap with class-level segmentation ---
seg_env = SegmentationMaskWrapper(env, seg_mode="class")

print("=== observation_space ===")
for k, v in seg_env.observation_space.spaces.items():
    print(f"  {k}: {v}")

# --- reset ---
obs = seg_env.reset()
seg_mask = obs["seg_mask"]
seg_labels = obs["seg_labels"]

print(f"\n=== after reset ===")
print(f"seg_mask shape : {seg_mask.shape}")
print(f"seg_mask dtype : {seg_mask.dtype}")
print(f"unique IDs     : {np.unique(seg_mask).tolist()}")
print(f"seg_labels     : {seg_labels}")
save_seg_image(obs, seg_labels, "reset_class.png", title="Class mode — reset")
for i in range(5):
    action = seg_env.action_space.sample()
    obs, reward, done, info = seg_env.step(action)
    seg_mask = obs["seg_mask"]
    seg_labels = obs["seg_labels"]
    print(f"\n=== step {i+1} (action={action}) ===")
    print(f"  unique IDs : {np.unique(seg_mask).tolist()}")
    print(f"  labels     : {seg_labels}")
    print(f"  reward={reward}  done={done}")
    save_seg_image(obs, seg_labels, f"step{i+1}_class.png",
                   title=f"Class mode — step {i+1} (action={action})")
    if done:
        obs = seg_env.reset()
        print("  (env reset)")

# --- also test instance mode ---
print("\n\n=== instance mode ===")
env2 = gym.make(
    "homegrid-task",
    disable_env_checker=True,
    max_steps=100,
    num_trashobjs=2,
    num_trashcans=2,
)
env2 = homegrid.wrappers.Gym26Wrapper(env2)
seg_env2 = SegmentationMaskWrapper(env2, seg_mode="instance")
obs2 = seg_env2.reset()
print(f"seg_mask shape : {obs2['seg_mask'].shape}")
print(f"unique IDs     : {np.unique(obs2['seg_mask']).tolist()}")
print(f"seg_labels     : {obs2['seg_labels']}")
save_seg_image(obs2, obs2["seg_labels"], "reset_instance.png",
               title="Instance mode — reset")
