#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Path to astar.py
ASTAR_PATH = Path("GroupProblem/True_Python/src/astar.py").resolve()

_print_lock = Lock()


def print_top5_steps_unperf0(path: str, top_n: int = 15) -> None:
    best = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
                continue

            if int(obj.get("unperf", 1)) != 0 or int(obj.get("steps", 1)) == 0:
                continue

            best.append(obj)

    best.sort(key=lambda x: int(x.get("steps", 10**18)))

    for i, obj in enumerate(best[:top_n], 1):
        print(
            f"{i}) steps={obj.get('steps')}, temp={obj.get('temp')}, "
            f"run={obj.get('run')}, dt_sec={obj.get('dt_sec')}, ts={obj.get('ts')}"
        )


def _run_one_temp(
    temp: float,
    time_limit: float,
    jsonl_file: str,
    data_path: str,
    timeout: float = 30.0,
) -> str:
    """One astar.py run for a specific TEMP. Returns stdout or error string."""
    try:
        cmd = [
            sys.executable,
            str(ASTAR_PATH),
            f"--temp={temp:.9f}",
            "--runs=1",
            f"--jsonl={jsonl_file}",
            f"--time_limit={time_limit}",
        ]

        # Pass DATA_path into astar.py if provided
        if data_path:
            cmd.append(f"--data_path={data_path}")  # argparse ожидает --data_path [web:187]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"ERROR (TEMP {temp:.6f}): {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT (TEMP {temp:.6f})"


def run_astar(
    temp_start: float,
    jsonl_file: str,
    data_path: str,
    num_temps: int = 100,
    step: float = 0.01,
    workers: int = 8,
    time_limit: float = 10.0,
    timeout: float = 30.0,
) -> None:
    """Runs astar.py in parallel for TEMP range using N workers."""
    temps = [temp_start + i * step for i in range(num_temps)]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run_one_temp, t, time_limit, jsonl_file, data_path, timeout): t
            for t in temps
        }

        for fut in as_completed(futures):
            out = fut.result()
            with _print_lock:
                print(out)


if __name__ == "__main__":
    NAME = "BIRDS_3"

    run_id = NAME.split("_")[-1] 
    DATA_path = "GroupProblem/True_Python/data/outputs/" + NAME + "/parsed_data.json"
    jsonl_file = f"runs_{run_id}.jsonl"
    top_n = 5   # Best count

    run_astar(
        temp_start=1.0,
        jsonl_file=jsonl_file,
        data_path=DATA_path,
        num_temps=1,
        step=0.001,
        workers=8,
        time_limit=5.0,
    )
    print_top5_steps_unperf0(jsonl_file, top_n=top_n)
