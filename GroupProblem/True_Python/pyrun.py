import json
import os
from os.path import join

from tests import run_e2e_optimize

if __name__ == "__main__":
    DATA_ROOT = "GroupProblem/True_Python/data"

    inputs_dir = os.path.join(DATA_ROOT, "inputs")
    output_dir = os.path.join(DATA_ROOT, "outputs")

    with open(os.path.join(DATA_ROOT, "test_config.json")) as fp:
        config = json.load(fp)

    # Run full pipeline
    run_e2e_optimize(
        inputs_dir=inputs_dir,
        output_dir=output_dir,
        config=config
    )
