"""Visualize step_low schedule with different configurations."""
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def step_low(start_epoch, cur_epoch, total_epoch, step_num_small, step_num_large, power=2.0):
    epoch_train = total_epoch - start_epoch
    if epoch_train <= 0:
        return step_num_small
    progress = min((cur_epoch - start_epoch) / epoch_train, 1.0)
    result = step_num_large - (step_num_large - step_num_small) * (progress ** power)
    return max(int(result) + 1, step_num_small)


def plot_schedule(ax, start_epoch, total_epoch, step_num_small, step_num_large, powers, title):
    epochs = list(range(start_epoch, total_epoch + 1))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(powers)))
    for power, color in zip(powers, colors):
        lbs = [step_low(start_epoch, e, total_epoch, step_num_small, step_num_large, power)
               for e in epochs]
        ax.step(epochs, lbs, where='post', label=f'power={power}', color=color, linewidth=2)

    ax.axhline(step_num_large, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label=f'large={step_num_large}')
    ax.axhline(step_num_small, color='gray', linestyle=':',  linewidth=0.8, alpha=0.5, label=f'small={step_num_small}')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lower bound for z_t_hat')
    ax.legend(fontsize=8)
    ax.set_yticks(range(step_num_small, step_num_large + 3))
    ax.grid(True, alpha=0.3)


# ── Configurations ────────────────────────────────────────────────────────────
configs = [
    dict(start_epoch=0, total_epoch=56, step_num_small=0, step_num_large=6,
         title='start=0, end=56, small=0, large=6'),
    dict(start_epoch=0,  total_epoch=146, step_num_small=0, step_num_large=8,
         title='start=0, end=146, small=0, large=8'),
    dict(start_epoch=0, total_epoch=106, step_num_small=0, step_num_large=8,
         title='start=0, end=106, small=0, large=8'),
    dict(start_epoch=5,  total_epoch=26, step_num_small=5, step_num_large=11,
         title='start=5, end=26, small=5, large=11'),
]

powers = [0.5, 0.75, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('step_low schedule — lower bound for z_t_hat sampling', fontsize=13)

for ax, cfg in zip(axes.flat, configs):
    plot_schedule(ax, powers=powers, **cfg)

plt.tight_layout()
plt.savefig('schedule.png', dpi=150, bbox_inches='tight')
print('Saved schedule.png')
plt.show()
