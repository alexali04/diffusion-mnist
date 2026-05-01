import argparse


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_sample_args():
    parser = argparse.ArgumentParser(description="Sample images from a trained diffusion model")
    parser.add_argument("checkpoint", type=str, help="Path to a model checkpoint saved by affine.py")
    parser.add_argument("--num_samples", type=positive_int, default=10)
    parser.add_argument("--num_steps", type=positive_int, default=50)
    parser.add_argument("--pred_param", choices=["x0", "eps"], default=None)
    parser.add_argument("--title", type=str, default="samples")
    parser.add_argument("--name", type=str, default="")

    return parser
