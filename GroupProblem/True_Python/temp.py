#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed  # [web:137]
from threading import Lock


# Путь к astar.py
ASTAR_PATH = Path("GroupProblem/True_Python/src/astar.py").resolve()

_print_lock = Lock()


def print_top5_steps_unperf0(path: str = "runs_3.jsonl") -> None:
    best = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Process obj here
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
            if int(obj.get("unperf", 1)) != 0 or int(obj.get("steps", 1)) == 0:
                continue
            best.append(obj)

    best.sort(key=lambda x: int(x.get("steps", 10**18)))

    for i, obj in enumerate(best[:15], 1):
        print(
            f"{i}) steps={obj.get('steps')}, temp={obj.get('temp')}, "
            f"run={obj.get('run')}, dt_sec={obj.get('dt_sec')}, ts={obj.get('ts')}"
        )


def _run_one_temp(temp: float, timeout: float = 30.0) -> str:
    """Один запуск astar.py для конкретного TEMP. Возвращает stdout (или строку ошибки)."""
    try:
        result = subprocess.run(
            [sys.executable, str(ASTAR_PATH), f"--temp={temp:.3f}", "--runs=1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"ERROR (TEMP {temp:.3f}): {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT (TEMP {temp:.3f})"


def run_astar(temp_start: float, num_temps: int = 100, step: float = 0.01, workers: int = 8):
    """Параллельно запускает astar.py для диапазона TEMP в N workers."""
    temps = [temp_start + i * step for i in range(num_temps)]

    with ThreadPoolExecutor(max_workers=workers) as ex:  # ThreadPoolExecutor подходит для subprocess [web:137]
        futures = {ex.submit(_run_one_temp, t): t for t in temps}

        for fut in as_completed(futures):
            out = fut.result()
            with _print_lock:
                print(out)


if __name__ == "__main__": # 6
    run_astar(temp_start=4.08, num_temps=1, step=0.001, workers=8)
    print_top5_steps_unperf0()
