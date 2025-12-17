#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path

# Путь к astar.py
ASTAR_PATH = Path("GroupProblem/True_Python/src/astar.py").resolve()

def print_top5_steps_unperf0(path: str = "runs.jsonl") -> None:
    best = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # JSONL: 1 JSON-объект на строку [web:124]
            if int(obj.get("unperf", 1)) != 0:
                continue
            best.append(obj)

    best.sort(key=lambda x: int(x.get("steps", 10**18)))

    for i, obj in enumerate(best[:4], 1):
        print(
            f"{i}) steps={obj.get('steps')}, temp={obj.get('temp')}, "
            f"run={obj.get('run')}, dt_sec={obj.get('dt_sec')}, ts={obj.get('ts')}"
        )

def run_astar(temp_start, num_temps=100, step=0.01):
    """Запускает astar.py для диапазона TEMP"""
    
    current_temp = temp_start
    best_overall_steps = float('inf')
    best_overall_temp = temp_start
    
    for i in range(num_temps):
        try:
            # Запуск astar.py с аргументами
            result = subprocess.run(
                [sys.executable, str(ASTAR_PATH), f"--temp={current_temp:.3f}", "--runs=1"],
                capture_output=True, 
                text=True, 
                timeout=30  # 30 сек таймаут на случай зависания
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                print(output)
                
                # Парсим результат для поиска лучшего
                # if "BEST: Temp" in output:
                #     lines = output.split('\n')
                #     for line in lines:
                #         if line.startswith("Temp") and "steps" in line:
                #             parts = line.split()
                #             steps = int(parts[parts.index('steps') - 1])
                #             temp_val = float(parts[1])
                #             if steps < best_overall_steps:
                #                 best_overall_steps = steps
                #                 best_overall_temp = temp_val
            else:
                print(f"ERROR (TEMP {current_temp:.3f}): {result.stderr}")
            
            current_temp += step
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT (TEMP {current_temp:.3f})")
            current_temp += step
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break


if __name__ == "__main__":
    # Запуск с вашими параметрами
    run_astar(temp_start=3, num_temps=10000, step=0.001)
    print_top5_steps_unperf0()
