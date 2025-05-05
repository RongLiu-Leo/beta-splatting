import os

import sys
from scene import BetaModel
from argparse import ArgumentParser

import time


def compressing(args):
    beta_model = BetaModel()
    beta_model.load_ply(args.ply)
    start_time = time.time()
    beta_model.save_png(os.path.dirname(args.ply))
    end_time = time.time()
    print(f"Compression time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Compressing script parameters")
    parser.add_argument(
        "--ply", type=str, default=None, help="path to the .ply file"
    )
    args = parser.parse_args(sys.argv[1:])

    print("Compressing " + args.ply)

    compressing(args)
