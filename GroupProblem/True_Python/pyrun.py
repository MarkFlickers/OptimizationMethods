import json
import sys
import os
import argparse
from os.path import join

from tests import run_e2e_optimize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full e2e optimization pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="GroupProblem/True_Python/data/test_config.json",
        help="Path to config JSON file (default: data/test_config.json)"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="GroupProblem/True_Python/data",
        help="Root folder for input/output data (default: GroupProblem/True_Python/data)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for moving checking (default: False)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = args.data_root
    print(f"Data root: {data_root}")

    inputs_dir = os.path.join(data_root, "inputs")
    output_dir = os.path.join(data_root, "outputs")

    config_path = args.config
    print(f"Config path: {config_path}")

    verbose = args.verbose

    error_found = False

    if not os.path.exists(config_path):
        print(f"\n\tConfig file not found: {config_path}")
        print("\tThe config must be created based on: data/test_config_template.json")
        error_found = True

    if not os.path.exists(data_root):
        print(f"\tData folder not found: {data_root}")
        error_found = True

    if error_found:
        print(f"\tCurrent working directory: {os.getcwd()}\n")
        sys.exit(0)

    with open(config_path, "r") as fp:
        config = json.load(fp)

    run_e2e_optimize(
        inputs_dir=inputs_dir,
        output_dir=output_dir,
        config=config,
        verbose=verbose
    )
