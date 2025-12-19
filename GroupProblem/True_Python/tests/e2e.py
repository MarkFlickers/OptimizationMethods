import time
import json
import sys
import os
from datetime import datetime
from os.path import join
from os import makedirs

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
import importlib

from src import (
    BranchProcessor,
    OrderProcessor,
    BranchIntegrity,
    AStarSolver,
    State,
    tune_temp,
)

def load_data_json_from_parsed(parsed_path: str) -> str:
    """
    parsed_data.json выглядит так:
    {"DATA": [...], "BRANCH_LEN": ..., "BIRDS_COUNT": ...}
    Нам нужен только DATA, и вернуть его в виде JSON-строки для --data_json.
    """
    with open(parsed_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload["DATA"]
    return json.dumps(data, ensure_ascii=False)

def choose_temp_from_matrix_or_tune(
    matrix_path: str,
    expected_steps_set: set[int],
    astar_py_path: str,
    data_path: str,
    tuner_py_path: str,
    tuner_out_json: str,
    tuner_log_path: str,
    workers: int = 8,
    time_limit: float = 3.0,
    tune_t_min: float = 1.0,
    tune_t_max: float = 10.0,
    tune_workers: int = 8,
) -> float:
    matrix_rows = load_matrix_config_jsonl(matrix_path)

    temps = []
    for r in matrix_rows:
        if "temp" in r:
            temps.append(float(r["temp"]))

    uniq = []
    seen = set()
    for t in temps:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - matrix_config: {matrix_path}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - matrix_config rows={len(matrix_rows)}, uniq_temps={len(uniq)}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - simulating variable from matrix_config with workers={workers}, time_limit={time_limit}s")

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_astar_subprocess_once, astar_py_path, t, data_path, time_limit): t
            for t in uniq
        }

        done = 0
        total = len(futs)

        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            done += 1

            if "error" in res:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - [{done}/{total}] variable={t} -> ERROR={res.get('error')}")
                if res.get("stderr"):
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - stderr: {res['stderr']}")
                if res.get("stdout"):
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - stdout: {res['stdout']}")
                continue

            steps = int(res.get("steps", 10**18))
            unperf = res.get("unperf", None)
            dt_sec = res.get("dt_sec", None)

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - [{done}/{total}] variable={t} -> steps={steps}, unperf={unperf}, dt={dt_sec}")

            if steps in expected_steps_set:
                dt_total = time.perf_counter() - t0
                chosen = float(res.get("temp", t))
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - MATCH FOUND: steps={steps} -> chosen variable={chosen} (sim_time={dt_total:.3f}s)")
                return chosen

    dt_total = time.perf_counter() - t0
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - no steps match found in matrix_config (sim_time={dt_total:.3f}s)")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - starting tuner subprocess...")

    tuned = run_tuner_subprocess(
        tuner_py_path=tuner_py_path,
        astar_py_path=astar_py_path,
        data_path=data_path,
        out_json_path=tuner_out_json,
        log_path=tuner_log_path,
        t_min=tune_t_min,
        t_max=tune_t_max,
        workers=tune_workers,
    )

    if "error" in tuned:
        raise RuntimeError(f"Tuner failed: {tuned}")

    chosen = float(tuned["temp"])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - tuner finished -> chosen TEMP={chosen}")
    return chosen


def load_matrix_config_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def run_tuner_subprocess(
    tuner_py_path: str,
    astar_py_path: str,
    data_path: str,
    out_json_path: str,
    log_path: str,
    t_min: float = 1.0,
    t_max: float = 10.0,
    workers: int = 8,
) -> dict:
    cmd = [
        sys.executable,
        tuner_py_path,
        f"--astar_path={astar_py_path}",
        f"--data_path={data_path}",
        f"--out_json={out_json_path}",
        f"--log_path={log_path}",
        f"--t_min={t_min}",
        f"--t_max={t_max}",
        f"--workers={workers}",
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return {
            "error": "tuner_returncode",
            "returncode": p.returncode,
            "stderr": p.stderr[-4000:],
            "stdout": p.stdout[-4000:],
        }

    # читаем out_json
    try:
        with open(out_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": "tuner_no_out_json", "exc": str(e), "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}



def run_astar_subprocess_once(astar_path: str, temp: float, data_path: str, time_limit: float = 3.0) -> dict:
    """
    Запускает astar.py и ожидает, что ПОСЛЕДНЯЯ строка stdout = JSON.
    DATA передаём через --data_path (parsed_data.json).
    """
    cmd = [
        sys.executable,
        astar_path,
        f"--temp={temp}",
        f"--time_limit={time_limit}",
        "--runs=1",
        "--jsonl=",
        f"--data_path={data_path}",
    ]

    # DATA = list()
    # with open(data_path, "r", encoding="utf-8") as f:
    #     payload = json.load(f)
    # DATA = payload["DATA"]
    # print("DATA_LEN", len(DATA), "ROW_LEN", len(DATA[0]) if DATA else 0, file=sys.stderr)

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 0.5,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    if p.returncode != 0:
        return {"error": "returncode", "stderr": p.stderr[-2000:], "stdout": p.stdout[-2000:]}

    lines = p.stdout.strip().splitlines()
    if not lines:
        return {"error": "no_stdout"}

    try:
        return json.loads(lines[-1])
    except Exception:
        return {"error": "bad_json", "stdout": p.stdout[-2000:]}



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
    def __init__(self, inputs: dict[str, str], output_dir: str, config: dict, verbose: bool = True):
        self.inputs = inputs              # {"file": "content"}
        self.output_dir = output_dir      # folder for results
        self.config = config              # parsed test_config.json
        self.data = {}                    # intermediate results: DATA, BRLEN, solution, etc.
        self.verbose = verbose

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
# Step 1.5 — Branch integrity check
# ============================================================================

class BranchIntegrityStep(PipelineStep):
    name = "branch_integrity"

    def run(self, ctx: PipelineContext):
        input_file = ctx.config["input_file"]
        full_path = os.path.join(ctx.output_dir, input_file)
        integrity = BranchIntegrity(full_path, ctx.verbose)
        integrity.run()

        if integrity.err:
            raise RuntimeError("BranchIntegrity failed, invalid data")

        # Сохраняем результаты в контекст, чтобы можно было при желании использовать
        ctx.data["BRANCH_INTEGRITY"] = {
            "branches": integrity.branches,
            "order": integrity.order,
            "BRNCHLEN": integrity.BRNCHLEN
        }

        return ctx

class TempPreflightStep(PipelineStep):
    name = "temp_preflight"

    def run(self, ctx: PipelineContext):
        """
        Перед solve:
        - читаем matrix_config.jsonl
        - запускаем astar.py на temps из него (параллельно), time_limit=3s
        - если нашли совпадение по steps с тем, что есть в matrix_config -> берём temp
        - иначе -> tune_temp и берём temp оттуда
        - сохраняем выбранный temp в ctx.data["TEMP"]
        """
        cfg = ctx.config

        matrix_path = cfg.get("matrix_config_file", "matrix_config.jsonl")
        astar_py = cfg.get("astar_py_path")  # ОБЯЗАТЕЛЬНО укажи в config
        workers = int(cfg.get("temp_workers", 8))
        time_limit = float(cfg.get("temp_time_limit", 3.0))

        # steps-цели из matrix_config
        rows = load_matrix_config_jsonl(matrix_path)
        expected_steps_set = set(int(r["steps"]) for r in rows if "steps" in r)

        # tune_temp берём как функцию (ты можешь импортировать её откуда нужно)
        # Вариант 1: если tune_temp находится в этом же файле - просто используй tune_temp напрямую.
        # Вариант 2: если tune_temp в другом модуле - импортируй тут.

        data_path = os.path.join(ctx.output_dir, "parsed_data.json")

        tuner_py = cfg.get("tuner_py_path")  # путь до tune_temp.py
        if not tuner_py:
            raise RuntimeError("Config must contain 'tuner_py_path' (path to tune_temp.py)")

        tuner_log_path = cfg.get("tune_log_path", os.path.join(ctx.output_dir, "tune_log.jsonl"))
        tuner_out_json = os.path.join(ctx.output_dir, "chosen_temp.json")

        t_min = float(cfg.get("tune_t_min", 1.0))
        t_max = float(cfg.get("tune_t_max", 10.0))
        tune_workers = int(cfg.get("tune_workers", workers))

        chosen_temp = choose_temp_from_matrix_or_tune(
            matrix_path=matrix_path,
            expected_steps_set=expected_steps_set,
            astar_py_path=astar_py,
            data_path=data_path,
            tuner_py_path=tuner_py,
            tuner_out_json=tuner_out_json,
            tuner_log_path=tuner_log_path,
            workers=workers,
            time_limit=time_limit,
            tune_t_min=t_min,
            tune_t_max=t_max,
            tune_workers=tune_workers,
        )
        ctx.data["TEMP"] = float(chosen_temp)
        ctx.save_json("chosen_temp.json", {"temp": chosen_temp})
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - chosen TEMP={chosen_temp:.6f}")

        return ctx


# ============================================================================
# Step 2 — Run A* solver
# ============================================================================


class SolveStep(PipelineStep):
    name = "solve"

    def run(self, ctx: PipelineContext):

        # 1. Берём текущие данные после parse/apply_order
        DATA = ctx.data["DATA"]

        # <<< ДОБАВИТЬ: применяем выбранный TEMP >>>
        if "TEMP" in ctx.data:
            # ВАРИАНТ: если TEMP живёт в src.astar (или другом модуле) — импортни и поменяй там
            import src.astar as astar_mod
            astar_mod.TEMP = float(ctx.data["TEMP"])

        # 2. Создаём объект состояния оптимизированного A*
        start_state = State.from_lists(DATA)

        # 3. Запускаем оптимизированный решатель
        solver = AStarSolver(start_state)
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
# Pipeline manager
# ============================================================================

class E2EPipeline:
    """
    Orchestrates any steps conditionally based on input.
    """
    def __init__(self, context: PipelineContext):
        self.ctx = context

        self.steps = {
            ParseStep.name: ParseStep(),
            BranchIntegrityStep.name: BranchIntegrityStep(),
            TempPreflightStep.name: TempPreflightStep(),
            SolveStep.name: SolveStep(),
        }

    def run_conditional(self):
        input_file = self.ctx.config["input_file"]
        lines = self.ctx.load_input_lines(input_file)

        # Проверяем, есть ли ORDER в исходном файле
        has_order = any(l.strip() and l.strip().upper() == "ORDER" for l in lines)

        # Список шагов, которые будут выполняться
        step_sequence = []

        if has_order:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - ORDER detected, running BranchIntegrity only")
            step_sequence.append("branch_integrity")
        else:
            step_sequence.extend(["parse", "temp_preflight", "solve"])

        for step_name in step_sequence:
            if step_name not in self.steps:
                raise ValueError(f"Unknown step: {step_name}")
            step_obj = self.steps[step_name]

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} = STEP = {step_name}")
            t0 = time.perf_counter()

            try:
                self.ctx = step_obj.run(self.ctx)
                if step_name == "parse":
                    brlen = int(self.ctx.data.get("BRANCH_LEN", 0))
                    if brlen > 26:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - branch too long (BRANCH_LEN={brlen} > 26), timeout: skipping temp_preflight and solve")
                        # опционально: сохраним маркер в output_dir
                        try:
                            self.ctx.save_json("timeout.json", {"reason": "branch_too_long", "BRANCH_LEN": brlen})
                        except Exception:
                            pass
                        return self.ctx
            except Exception as e:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Step '{step_name}' failed: {e}")
                raise

            t1 = time.perf_counter()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - [{step_name} time]: {t1 - t0:.4f} sec")

        # После решения — проверяем BranchIntegrity на результат, если выполнялся solve
        if "solve" in step_sequence:
            solution_file = os.path.join(self.ctx.output_dir, input_file)
            moves = self.ctx.data["solution"]["moves"]

            data_text = "\n".join(lines).strip()
            order_lines = []
            for mv in moves:
                s = mv["src_branch"] + 1
                d = mv["dst_branch"] + 1
                b = chr(mv["bird"] + ord("A") - 1)
                order_lines.append(f"{s} {d} {b}")

            full_text = data_text + "\n\nORDER\n" + "\n".join(order_lines) + "\n/"
            with open(solution_file, "w") as fp:
                fp.write(full_text)

            # Запускаем BranchIntegrity на выходной файл
            step_name = "branch_integrity"
            step_obj = self.steps[step_name]
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} = STEP = {step_name} (after solve)")
            t0 = time.perf_counter()
            self.ctx = step_obj.run(self.ctx)
            t1 = time.perf_counter()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - [{step_name} time]: {t1 - t0:.4f} sec")

        return self.ctx


def run_e2e(
    inputs_dir: str,
    output_dir: str,
    config: dict,
    verbose: bool,
    # steps: list[str] | None = None
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
        ctx = PipelineContext(inputs_map, test_output_dir, local_config, verbose)
        pipeline = E2EPipeline(ctx)

        # steps берём из config, если не передано параметром
        # step_sequence = steps if steps is not None else config.get("steps", None)

        try:
            pipeline.run_conditional()
        except RuntimeError:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Pipeline stopped due to error.")
            # просто прекращаем без traceback
            return

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
