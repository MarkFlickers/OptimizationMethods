import json
import os
from os.path import join
from os import makedirs

from src import (
    BranchProcessor,
    OrderProcessor,
    TreeState,
    AStarSolver,
)

# ============================================================================
# Base class for all steps
# ============================================================================

class PipelineStep:
    """Abstract pipeline step."""
    name = "base_step"

    def run(self, ctx):
        raise NotImplementedError


# ============================================================================
# Context object shared between steps
# ============================================================================

class PipelineContext:
    """
    Stores:
      - input files content
      - configuration
      - parsed structures
      - output directory
    """
    def __init__(self, inputs: dict[str, str], output_dir: str, config: dict):
        self.inputs = inputs              # {"file": "content"}
        self.output_dir = output_dir      # folder for results
        self.config = config              # parsed test_config.json
        self.data = {}                    # intermediate results: DATA, BRLEN, solution, etc.

        makedirs(output_dir, exist_ok=True)

    def save_json(self, name, payload):
        path = join(self.output_dir, name)
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2)
        return path

    def load_input_lines(self, file_name):
        return self.inputs[file_name].splitlines()


# ============================================================================
# Step 1 — Parse input
# ============================================================================

class ParseStep(PipelineStep):
    name = "parse"

    def run(self, ctx: PipelineContext):
        print("[step] PARSE")

        input_file = ctx.config["input_file"]
        lines = ctx.load_input_lines(input_file)

        bp = BranchProcessor(lines)
        start_data_idx, end_data_idx, branch_count = bp.validdata()
        err, DATA, BRLEN, CNT = bp.process_branches(start_data_idx, end_data_idx)

        if err != 0:
            raise RuntimeError(f"Parsing error: code={err}")

        ctx.data["DATA"] = DATA
        ctx.data["BRANCH_LEN"] = BRLEN
        ctx.data["BIRDS_COUNT"] = CNT

        ctx.save_json("parsed_data.json", {
            "DATA": DATA,
            "BRANCH_LEN": BRLEN,
            "BIRDS_COUNT": CNT,
        })

        return ctx


# ============================================================================
# Step 2 — Apply ORDER section
# ============================================================================

class ApplyOrderStep(PipelineStep):
    name = "apply_order"

    def run(self, ctx: PipelineContext):
        print("[step] APPLY ORDER")

        DATA = ctx.data["DATA"]
        BRLEN = ctx.data["BRANCH_LEN"]

        input_file = ctx.config["input_file"]
        lines = ctx.load_input_lines(input_file)

        op = OrderProcessor(lines, BRLEN)
        order_start, rel_end = op.find_order_section()

        if order_start != -1 and rel_end != -1:
            order_lines = lines[order_start+1: order_start+rel_end]
            moves = [l.split() for l in order_lines if l.strip()]
        else:
            moves = []

        # Apply moves
        for i, m in enumerate(moves):
            src = int(m[0]) - 1
            dst = int(m[1]) - 1
            bird_char = m[2]
            bird_num = ord(bird_char) - ord('A') + 1

            if len(DATA[src]) == 0:
                raise RuntimeError("ORDER: source empty")

            if DATA[src][-1] != bird_num:
                raise RuntimeError("ORDER: wrong bird on source")

            if len(DATA[dst]) >= BRLEN:
                raise RuntimeError("ORDER: destination full")

            if len(DATA[dst]) > 0 and DATA[dst][-1] != bird_num:
                raise RuntimeError("ORDER: stack mismatch")

            DATA[dst].append(DATA[src].pop())

        ctx.data["DATA"] = DATA

        ctx.save_json("order_applied.json", {"DATA_AFTER_ORDER": DATA})
        return ctx


# ============================================================================
# Step 3 — Run A* solver
# ============================================================================

class SolveStep(PipelineStep):
    name = "solve"

    def run(self, ctx: PipelineContext):
        print("[step] SOLVE")

        DATA = ctx.data["DATA"]
        BRLEN = ctx.data["BRANCH_LEN"]

        # Convert to full rectangular matrix
        matrix = []
        for row in DATA:
            r = list(row)
            if len(r) < BRLEN:
                r += [0] * (BRLEN - len(r))
            matrix.append(r)

        ts = TreeState(matrix)
        solver = AStarSolver(ts)
        solution = solver.solve()

        payload = {
            "steps": solution.steps_amount,
            "moves": [m.__dict__ for m in solution.Moves],
            "result_tree": solution.Resultant_tree,
        }
        ctx.save_json("solution.json", payload)

        ctx.data["solution"] = payload
        return ctx


# ============================================================================
# Step 4 — Verify final state
# ============================================================================

class VerifyStep(PipelineStep):
    name = "verify"

    def run(self, ctx: PipelineContext):
        print("[step] VERIFY")

        solution = ctx.data["solution"]
        resultant = solution["result_tree"]

        for br in resultant:
            nonzero = [x for x in br if x != 0]
            if not nonzero:
                continue
            if any(x != nonzero[0] for x in nonzero):
                raise RuntimeError("Verification failed: branch not uniform")

        print("Verified: OK")
        return ctx


# ============================================================================
# Pipeline manager
# ============================================================================

class E2EPipeline:
    """
    Orchestrates any steps:
        pipeline.run(["parse", "solve"])
        pipeline.run_all()
    """
    def __init__(self, context: PipelineContext):
        self.ctx = context

        self.steps = {
            ParseStep.name: ParseStep(),
            ApplyOrderStep.name: ApplyOrderStep(),
            SolveStep.name: SolveStep(),
            VerifyStep.name: VerifyStep(),
        }

    def run(self, step_list):
        """Run specific steps in order."""
        for s in step_list:
            if s not in self.steps:
                raise ValueError(f"Unknown step: {s}")
            self.ctx = self.steps[s].run(self.ctx)
        return self.ctx

    def run_all(self):
        """Standard pipeline: parse → apply_order → solve → verify"""
        return self.run(["parse", "apply_order", "solve", "verify"])

    def validate_only(self):
        """Only parse + verify (if solution exists)."""
        return self.run(["parse", "apply_order", "verify"])


# ============================================================================
# Convenient top-level function
# ============================================================================

def run_e2e(
    inputs_dir: str,
    output_dir: str,
    config: dict,
    steps: list[str] | None = None
):
    input_files = config.get("input_files", [])
    if not input_files:
        raise RuntimeError("Config error: 'input_files' list is empty")

    # Перебираем все входные файлы из config.json
    for fname in input_files:
        input_path = os.path.join(inputs_dir, fname)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # создаём поддиректорию для каждого входного файла
        test_name = os.path.splitext(fname)[0]
        test_output_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_output_dir, exist_ok=True)

        # читаем входной файл (строки)
        with open(input_path, "r") as fp:
            content = fp.read()

        # формируем inputs dict: { file_name → content }
        inputs_map = {fname: content}

        # правим config на конкретный input_file
        local_config = dict(config)
        local_config["input_file"] = fname

        # создаём контекст и пайплайн
        ctx = PipelineContext(inputs_map, test_output_dir, local_config)
        pipeline = E2EPipeline(ctx)

        # steps берём из config, если не передано параметром
        step_sequence = steps if steps is not None else config.get("steps", None)

        if step_sequence is None:
            pipeline.run_all()
        else:
            pipeline.run(step_sequence)

        print(f"Completed: {fname} -> {test_output_dir}")
