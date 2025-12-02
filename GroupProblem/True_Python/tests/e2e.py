#!/usr/bin/env python3
import os
import json
import subprocess
from os.path import dirname, join
from os import makedirs

from src import BranchProcessor, OrderProcessor
from src import TreeState, AStarSolver

# -----------------------------------------------------------------------------
# This file was created and refactored with the assistance of ChatGPT (OpenAI).
# Original logic, algorithms and intent were preserved while improving structure,
# readability and adherence to SOLID principles.
#
# The author of the project retains all rights to the original idea, logic and
# specifications. ChatGPT is a tool and does not claim authorship or copyright.
#
# You are free to use, modify and distribute this file as part of your project.
# -----------------------------------------------------------------------------

def save_state_if_enabled(save_state, state, state_path, key, value):
    if save_state:
        state[key] = value
        with open(state_path, "w") as fp:
            json.dump(state, fp, indent=2)

def load_state_value(save_state, state, key, default):
    if save_state:
        return state.get(key, default)
    return default

def check_step(state, step):
    return state["last_step"] is None or state["last_step"] == step


def run_e2e(test_name: str, data_root: str, save_state: bool = False):
    """
    test_name  — имя теста, папка в data_root/tests/<test_name>
    data_root  — корневая папка проекта, где лежат data/tests
    """

    tests_dir = join(data_root, "tests")
    test_dir = join(tests_dir, test_name)
    run_dir = join(tests_dir, "run_" + test_name)

    makedirs(run_dir, exist_ok=True)

    # ----------------------- load config --------------------------------------
    config_path = join(test_dir, "test_config.json")
    with open(config_path, "r") as fp:
        config = json.load(fp)

    input_text_path = join(test_dir, config["input_file"])
    with open(input_text_path, "r") as fp:
        lines = [x.strip() for x in fp.readlines()]

    # ----------------------- load / init state.json ---------------------------
    state_path = join(run_dir, "state.json")
    if save_state and os.path.exists(state_path):
        with open(state_path, "r") as fp:
            state = json.load(fp)
    else:
        state = {"last_step": None}
        if save_state:
            with open(state_path, "w") as fp:
                json.dump(state, fp)

    disabled = set(config.get("disabled_steps", []))

    # ======================= STEP 1 — PARSE DATA ===============================
    if "parse" not in disabled and check_step(state, "parse"):
        print("[step] parse input")

        save_state_if_enabled(save_state, state, state_path, "last_step", "parse")

        bp = BranchProcessor(lines)
        start_data_idx, end_data_idx, branch_count = bp.validdata()
        err, DATA, BRLEN, CNT = bp.process_branches(start_data_idx, end_data_idx)

        if err != 0:
            raise RuntimeError(f"Parsing error: code={err}")

        parsed_path = join(run_dir, "parsed_data.json")
        with open(parsed_path, "w") as fp:
            json.dump({
                "DATA": DATA,
                "BRANCH_LEN": BRLEN,
                "BIRDS_COUNT": CNT,
            }, fp, indent=2)

        save_state_if_enabled(save_state, state, state_path, "parsed_data", parsed_path)
        save_state_if_enabled(save_state, state, state_path, "branch_len", BRLEN)
        save_state_if_enabled(save_state, state, state_path, "DATA", DATA)
        save_state_if_enabled(save_state, state, state_path, "last_step", None)

    # ======================= STEP 2 — APPLY ORDER ==============================
    if "apply_order" not in disabled and check_step(state, "apply_order"):
        print("[step] apply ORDER section")

        save_state_if_enabled(save_state, state, state_path, "last_step", "apply_order")

        DATA = load_state_value(save_state, state, "DATA", None)
        BRLEN = load_state_value(save_state, state, "branch_len", None)

        op = OrderProcessor(lines, BRLEN)
        order_start, rel_end = op.find_order_section()

        if order_start != -1 and rel_end != -1:
            order_lines = lines[order_start+1: order_start+rel_end]
            moves = [l.split() for l in order_lines if l.strip()]
            # each move is [from, to, BIRDCHAR]
        else:
            moves = []

        # replay order
        for i, m in enumerate(moves):
            f = int(m[0]) - 1
            t = int(m[1]) - 1
            b = m[2]
            print(f"  applying order move #{i}: {m}")

            bird_num = ord(b) - ord('A') + 1

            # basic check: top bird matches
            if len(DATA[f]) == 0:
                raise RuntimeError("ORDER move: source empty")
            if DATA[f][-1] != bird_num:
                raise RuntimeError("ORDER move: wrong bird on source")

            # check destination
            if len(DATA[t]) >= BRLEN:
                raise RuntimeError("ORDER move: dest full")
            if len(DATA[t]) > 0 and DATA[t][-1] != bird_num:
                raise RuntimeError("ORDER move: cannot stack different birds")

            DATA[t].append(DATA[f].pop())

        order_applied_path = join(run_dir, "order_applied.json")
        with open(order_applied_path, "w") as fp:
            json.dump({"DATA_AFTER_ORDER": DATA}, fp, indent=2)

        save_state_if_enabled(save_state, state, state_path, "DATA", DATA)
        save_state_if_enabled(save_state, state, state_path, "order_applied", order_applied_path)
        save_state_if_enabled(save_state, state, state_path, "last_step", None)

    # ======================= STEP 3 — RUN A* SEARCH ============================
    if "solve" not in disabled and check_step(state, "solve"):
        print("[step] run A* solver")

        save_state_if_enabled(save_state, state, state_path, "last_step", "solve")

        DATA = load_state_value(save_state, state, "DATA", None)

        state_matrix = []
        for row in DATA:
            # pad to same length
            r = list(row)
            if len(r) < len(DATA[0]):
                r += [0] * (len(DATA[0]) - len(r))
            state_matrix.append(r)

        ts = TreeState(state_matrix)
        solver = AStarSolver(ts)
        solution = solver.solve()

        solution_path = join(run_dir, "solution.json")
        with open(solution_path, "w") as fp:
            json.dump({
                "steps": solution.steps_amount,
                "moves": [m.__dict__ for m in solution.Moves],
                "result_tree": solution.Resultant_tree,
            }, fp, indent=2)

        save_state_if_enabled(save_state, state, state_path, "solution", solution_path)
        save_state_if_enabled(save_state, state, state_path, "last_step", None)

    # ======================= STEP 4 — VERIFY RESULT ============================
    if "verify" not in disabled and check_step(state, "verify"):
        print("[step] verify final tree")

        save_state_if_enabled(save_state, state, state_path, "last_step", "verify")

        solution_path = load_state_value(save_state, state, "solution", None)
        with open(solution_path, "r") as fp:
            sol = json.load(fp)

        resultant = sol["result_tree"]

        # simple validation: every branch uniform or empty
        for br in resultant:
            nonzero = [x for x in br if x != 0]
            if len(nonzero) == 0:
                continue
            if any(x != nonzero[0] for x in nonzero):
                raise RuntimeError("Verification error: branch not uniform")

        print("Verified: OK")

        save_state_if_enabled(save_state, state, state_path, "last_step", None)

    print("E2E pipeline completed successfully!")
