import torch
import random
from datetime import datetime
from uuid import uuid4
import matplotlib.pyplot as plt


ADJECTIVES = [
    "bright", "quiet", "rapid", "gentle", "bold", "lucky", "silver", "cosmic"
]
NOUNS = [
    "river", "panda", "falcon", "forest", "signal", "matrix", "comet", "pixel"
]


def generate_run_id(args):
    if args.name != "":
        return args.name

    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    suffix = uuid4().hex[:6]
    return f"{adjective}-{noun}-{suffix}"


def log_diffuse(param_count, args, scheduler, run_id):
    run_date = datetime.now().astimezone().isoformat(timespec="seconds")

    log_str = (
        f"Date: {run_date}\n"
        f"Run `{run_id}`: Training DiT Transformer ({param_count:,} params) with learning rate {args.lr} "
        f"for {args.epochs} epochs using AdamW optimizer on MNIST. Currently logging\n"
        f"MSE over ELBO, diffusing for max {scheduler.config.num_train_timesteps} timesteps "
        f"(discrete time). Using {args.pred_param} parameterization.\n\n"
        "Additional Notes (add post-run):\n"
    )

    return log_str


def fwd_noise(x0, scheduler):
    noise = torch.randn_like(x0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.size(0),), device=x0.device).long()
    x_t = scheduler.add_noise(x0, noise, t)

    return noise, t, x_t


@torch.no_grad()
def sample(model, scheduler, labels, num_steps=50, device=torch.device("cpu")):
    model.eval()
    scheduler.set_timesteps(num_steps)
    n = labels.size(0)
    x = torch.randn(n, 1, 32, 32, device=device)
    for t in scheduler.timesteps:
        t_batch = t.expand(n).to(device)
        pred_x0 = model(x, timestep=t_batch, class_labels=labels.to(device)).sample
        x = scheduler.step(pred_x0, t, x).prev_sample
    return x.clamp(-1, 1)


@torch.no_grad()
def denoise_from(model, scheduler, x_T, labels, num_steps=50):
    model.eval()
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    for t in scheduler.timesteps:
        t_batch = t.expand(x.size(0)).to(x.device)
        eps = model(x, timestep=t_batch, class_labels=labels).sample
        x = scheduler.step(eps, t, x).prev_sample
    return x.clamp(-1, 1)

def test_denoising_ability(test_ds, device, scheduler, model, run_id, title):
    fixed_x0, fixed_y = [], []
    seen = set()
    for img, lbl in test_ds:
        if lbl not in seen:
            fixed_x0.append(img); fixed_y.append(lbl); seen.add(lbl)
        if len(seen) == 10:
            break
    fixed_x0 = torch.stack(fixed_x0).to(device)             # (10, 1, 32, 32)
    fixed_y  = torch.tensor(fixed_y, device=device)         # (10,)

    # Take the same 10 images, noise them at t=T, then denoise
    T_corrupt = 500
    torch.manual_seed(42)  # match the seed used in the pre-training viz
    t = torch.full((10,), T_corrupt, dtype=torch.long, device=device)
    noise = torch.randn_like(fixed_x0)
    x_T = scheduler.add_noise(fixed_x0, noise, t)
    x_recovered = denoise_from(model, scheduler, x_T, fixed_y)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
    for col in range(10):
        axes[0, col].imshow(fixed_x0[col, 0].cpu(),    cmap='gray', vmin=-1, vmax=1);   axes[0, col].axis('off')
        axes[1, col].imshow(x_T[col, 0].cpu(),         cmap='gray', vmin=-1.5, vmax=1.5); axes[1, col].axis('off')
        axes[2, col].imshow(x_recovered[col, 0].cpu(), cmap='gray', vmin=-1, vmax=1);   axes[2, col].axis('off')
    axes[0, 0].set_ylabel('original',  rotation=0, labelpad=30, fontsize=10)
    axes[1, 0].set_ylabel(f'noised\n(t={T_corrupt})', rotation=0, labelpad=35, fontsize=10)
    axes[2, 0].set_ylabel('denoised',  rotation=0, labelpad=30, fontsize=10)
    plt.tight_layout()
    plt.show()

    fig.savefig(f"img/{run_id}_denoising_{title}")
