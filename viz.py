import matplotlib.pyplot as plt
import math

def plot_losses(losses, ylabel, title, run_id):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(losses)
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(f"img/{run_id}_loss_{title}")
    plt.show()
    plt.close(fig)


def plot_samples(samples, title, run_id, labels=None):
    num_samples = samples.size(0)
    cols = min(10, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 2 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= num_samples:
            ax.axis('off')
            continue

        ax.imshow(samples[i, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
        if labels is not None:
            ax.set_title(str(labels[i].item() if hasattr(labels[i], "item") else labels[i]))
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(f"img/{run_id}_samples_{title}")
    plt.show()
    plt.close(fig)


def plot_ten_samples(samples, title, run_id):
    plot_samples(samples[:10], title, run_id, labels=range(10))