import os

import torch
from diffusers import DiTTransformer2DModel, DDPMScheduler

from const import model_kwargs, prediction_translation
from sample_parser import parse_sample_args
from utils import sample
from viz import plot_samples


def main(args):
    os.makedirs("img", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_model_kwargs = checkpoint.get("model_kwargs", model_kwargs)
    model = DiTTransformer2DModel(**checkpoint_model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_param = args.pred_param or checkpoint.get("pred_param", "x0")
    pred_type = prediction_translation.get(pred_param, "sample")
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type=pred_type)

    labels = torch.arange(args.num_samples, device=device) % 10
    samples = sample(model, scheduler, labels, num_steps=args.num_steps, device=device)

    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    run_id = args.name or checkpoint.get("run_id", checkpoint_name)
    plot_samples(samples, args.title, run_id, labels=labels.cpu())
    print(f"saved samples to img/{run_id}_samples_{args.title}")


if __name__ == "__main__":
    sample_parser = parse_sample_args()
    args = sample_parser.parse_args()
    main(args)
