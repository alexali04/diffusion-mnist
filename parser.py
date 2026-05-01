import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Options")
    parser.add_argument("--pred_param", choices=["x0", "eps"], default="x0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", type=str, default="")

    return parser