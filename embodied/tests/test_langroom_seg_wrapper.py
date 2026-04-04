"""Quick smoke-test for LangRoom segmentation."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from embodied.envs.langroom import LangRoom

OUT_DIR = "seg_test_images_langroom"
os.makedirs(OUT_DIR, exist_ok=True)


def save_seg_image(obs, seg_labels, filename, title=""):
    """Save side-by-side RGB image and colorized segmentation mask."""
    seg_mask = obs["seg_mask"]
    rgb = obs["image"]

    unique_ids = np.unique(seg_mask)
    n_ids = max(unique_ids.max() + 1, 2)
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


# --- class mode ---
print("=== class mode ===")
env = LangRoom(task="answer-only", seed=42, seg_mode="class")

print("obs_space:")
for k, v in env.obs_space.items():
    print(f"  {k}: {v}")

# Reset (step with reset action)
act = {k: v.sample() for k, v in env.act_space.items()}
act['reset'] = True
obs = env.step(act)
seg_mask = obs["seg_mask"]
seg_labels = obs["seg_labels"]

print(f"\nafter reset:")
print(f"  seg_mask shape : {seg_mask.shape}")
print(f"  seg_mask dtype : {seg_mask.dtype}")
print(f"  unique IDs     : {np.unique(seg_mask).tolist()}")
print(f"  seg_labels     : {seg_labels}")
save_seg_image(obs, seg_labels, "reset_class.png", title="Class mode — reset")

# Steps
for i in range(5):
    act = {k: v.sample() for k, v in env.act_space.items()}
    act['reset'] = False
    obs = env.step(act)
    seg_mask = obs["seg_mask"]
    seg_labels = obs["seg_labels"]
    print(f"\nstep {i+1}:")
    print(f"  unique IDs : {np.unique(seg_mask).tolist()}")
    print(f"  labels     : {seg_labels}")
    save_seg_image(obs, seg_labels, f"step{i+1}_class.png",
                   title=f"Class mode — step {i + 1}")

# --- instance mode ---
print("\n\n=== instance mode ===")
env2 = LangRoom(task="answer-only", seed=123, seg_mode="instance")
act = {k: v.sample() for k, v in env2.act_space.items()}
act['reset'] = True
obs2 = env2.step(act)
print(f"seg_mask shape : {obs2['seg_mask'].shape}")
print(f"unique IDs     : {np.unique(obs2['seg_mask']).tolist()}")
print(f"seg_labels     : {obs2['seg_labels']}")
save_seg_image(obs2, obs2["seg_labels"], "reset_instance.png",
               title="Instance mode — reset")

# --- no segmentation ---
print("\n\n=== no segmentation (default) ===")
env3 = LangRoom(task="answer-only", seed=0)
act = {k: v.sample() for k, v in env3.act_space.items()}
act['reset'] = True
obs3 = env3.step(act)
assert "seg_mask" not in obs3, "seg_mask should not be in obs when seg_mode='none'"
print("seg_mask correctly absent from obs")

print("\nAll tests passed!")
