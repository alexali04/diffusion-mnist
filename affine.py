import os
import time
import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel, DDPMScheduler

from parser import parse_args

from data import load_data, construct_config
from const import prediction_translation, model_kwargs
from utils import log_diffuse, fwd_noise, sample, generate_run_id, test_denoising_ability
from viz import plot_ten_samples, plot_losses

def main(args):
    os.makedirs("img", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(0)
    epochs, lr, pred_type = args.epochs, args.lr, prediction_translation.get(args.pred_param, "sample")
    run_id = generate_run_id(args)

    data_config = construct_config(args)
    train_dl, _, _, test_ds = load_data(data_config)

    model = DiTTransformer2DModel(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type=pred_type)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    log = log_diffuse(param_count, args, scheduler, run_id)

    # Training Loop
    losses = []
    model.train()
    start_time = time.time()
    for e in range(epochs):
        for step, (x0, y) in enumerate(train_dl):
            x0, y = x0.to(device), y.to(device)

            # forward process
            noise, t, x_t = fwd_noise(x0, scheduler)
            pred = model(x_t, timestep=t, class_labels=y).sample

            loss = F.mse_loss(pred, x0) if args.pred_param == "x0" else F.mse_loss(pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            losses.append(loss.item())
            if step % 50 == 0:
                print(f'epoch {e} step {step}: loss={loss.item():.4f}')

    total_time = time.time() - start_time
    final_loss = losses[-1] if losses else float("nan")

    log += (
        f"Total training time: {total_time:.2f} seconds\n"
        f"Final loss: {final_loss:.6f}\n\n"
    )
    print(log)

    with open("runs.log", "a", encoding="utf-8") as f:
        f.write(log)

    plot_losses(losses, ylabel="MSE", title="train_MSE", run_id=run_id)

    labels = torch.arange(10)
    samples = sample(model, scheduler, labels, num_steps=50, device=device)
    plot_ten_samples(samples, "samples", run_id)

    test_denoising_ability(test_ds, device, scheduler, model, run_id, "denoising")

    checkpoint_path = f"models/{run_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "pred_param": args.pred_param,
            "run_id": run_id,
        },
        checkpoint_path,
    )
    print(f"saved model to {checkpoint_path}")


if __name__ == "__main__":
    diffusion_parser = parse_args()
    args = diffusion_parser.parse_args()
    main(args)