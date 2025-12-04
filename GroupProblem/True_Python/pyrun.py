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
        default=None,
        help="Path to config JSON file (default: data/test_config.json)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DATA_ROOT = "GroupProblem/True_Python/data"

    inputs_dir = os.path.join(DATA_ROOT, "inputs")
    output_dir = os.path.join(DATA_ROOT, "outputs")

    config_path = args.config

    if config_path is None or not os.path.exists(config_path):
        print(
            f"\tConfig file not found: {config_path}\n"
            f"\tYou must provide a valid test configuration JSON.\n"
            f"\tUse argument: --config path/to/test_config.json\n"
            f"\tThe config must be created based on: data/test_config_template.json"
        )
        sys.exit(0)

    with open(config_path, "r") as fp:
        config = json.load(fp)

    run_e2e_optimize(
        inputs_dir=inputs_dir,
        output_dir=output_dir,
        config=config
    )
