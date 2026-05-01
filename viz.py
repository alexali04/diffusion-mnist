import matplotlib.pyplot as plt

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


def plot_ten_samples(samples, title, run_id):
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(samples[i, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(str(i)); axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(f"img/{run_id}_samples_{title}")
    plt.show()
    plt.close(fig)