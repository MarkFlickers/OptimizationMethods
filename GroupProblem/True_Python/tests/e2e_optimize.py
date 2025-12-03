
# -----------------------------------------------------------------------------
# This file was created and refactored with the assistance of ChatGPT (OpenAI).
# Logic, algorithms and semantics of the original C++ project have been preserved.
# -----------------------------------------------------------------------------

import time
import json
import sys
import os
from datetime import datetime
from os.path import join
from os import makedirs

from src import (
    BranchProcessor,
    OrderProcessor,
    TreeState,
    AStarSolver,
    AStarSolverOptimized,
    State
)

class TeeLogger:
    def __init__(self, logfile):
        self.log = open(logfile, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

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

        input_file = ctx.config["input_file"]
        lines = ctx.load_input_lines(input_file)

        bp = BranchProcessor(lines)
        start_data_idx, end_data_idx, branch_count = bp.validdata()
        err, DATA, BRLEN, CNT = bp.process_branches(start_data_idx, end_data_idx)

        if err != 0:
            raise RuntimeError(f"Parsing error: code={err}")

        ctx.data["DATA"] = DATA
        ctx.data["STATE"] = State.from_lists(DATA)
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
        ctx.data["STATE"] = State.from_lists(DATA)

        ctx.save_json("order_applied.json", {"DATA_AFTER_ORDER": DATA})
        return ctx


# ============================================================================
# Step 3 — Run A* solver
# ============================================================================


class SolveStep(PipelineStep):
    name = "solve"

    def run(self, ctx: PipelineContext):

        # 1. Берём текущие данные после parse/apply_order
        DATA = ctx.data["DATA"]

        # 2. Создаём объект состояния оптимизированного A*
        start_state = State.from_lists(DATA)

        # 3. Запускаем оптимизированный решатель
        solver = AStarSolverOptimized(start_state)
        steps_count, moves, result_state = solver.solve()

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {steps_count} steps")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - solve matrix:")

        for i, row in enumerate(result_state.branches):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - branch [{i+1:02d}] -", row)
        
        # 4. Формируем выходной JSON
        payload = {
            "steps": steps_count,
            "moves": [m.__dict__ for m in moves],
            "result_tree": [list(br) for br in result_state.branches],
        }

        # 5. Сохраняем файл solution.json
        ctx.save_json("solution.json", payload)

        # 6. Сохраняем результат в контекст
        ctx.data["solution"] = payload

        return ctx


# ============================================================================
# Step 4 — Verify final state
# ============================================================================

class VerifyStep(PipelineStep):
    name = "verify"

    def run(self, ctx: PipelineContext):

        solution = ctx.data["solution"]
        resultant = solution["result_tree"]

        for br in resultant:
            nonzero = [x for x in br if x != 0]
            if not nonzero:
                continue
            if any(x != nonzero[0] for x in nonzero):
                raise RuntimeError("Verification failed: branch not uniform")

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Verified: OK")
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
            step_obj = self.steps[s]
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} = STEP = {s}")

            t0 = time.perf_counter()
            self.ctx = step_obj.run(self.ctx)
            t1 = time.perf_counter()

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - [{s} time]: {t1 - t0:.4f} sec")
        return self.ctx

    def run_all(self):
        """Standard pipeline: parse -> apply_order -> solve -> verify"""
        return self.run(["parse", "apply_order", "solve", "verify"])

    def validate_only(self):
        """Only parse + verify (if solution exists)."""
        return self.run(["parse", "apply_order", "verify"])


# ============================================================================
# Convenient top-level function
# ============================================================================

def run_e2e_optimize(
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

        # LOGGER
        log_path = os.path.join(test_output_dir, f"{test_name}.log")
        sys.stdout = TeeLogger(log_path)

        # читаем входной файл (строки)
        with open(input_path, "r") as fp:
            content = fp.read()

        # Сохраняем копию входного файла в Outputs с тем же именем, но .txt
        txt_copy_path = os.path.join(test_output_dir, f"{test_name}.txt")
        with open(txt_copy_path, "w") as fp:
            fp.write(content)

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

        # 1. Берём исходный DATA (как текст)
        data_text = content.strip()

        # 2. Собираем ORDER из solution
        sol = ctx.data.get("solution")
        order_lines = []

        if sol is not None:
            for mv in sol["moves"]:
                s = mv["src_branch"] + 1
                d = mv["dst_branch"] + 1
                b = chr(mv["bird"] + ord("A") - 1)
                order_lines.append(f"{s} {d} {b}")

        # 3. Собираем текст
        full_text = data_text + "\n\nORDER\n"
        full_text += "\n".join(order_lines)
        full_text += "\n/"

        # 4. Сохраняем
        txt_output_path = os.path.join(test_output_dir, f"{test_name}.txt")
        with open(txt_output_path, "w") as fp:
            fp.write(full_text)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Completed: saved {fname} to {test_output_dir}")
        sys.stdout = sys.__stdout__
