#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Глобальные настройки уточнения
# -----------------------------
# 0) стартовая сетка (grid_n точек равномерно на [t_min, t_max])
# 1..5) 5 шагов уточнения: строим сетку вокруг лучшего temp:
#       [center-half_width, center+half_width] с шагом step, с clamp к [t_min, t_max]
REFINE_SCHEDULE = [
    {"name": "init_grid", "grid_n": 250, "time_limit": 5.0},
    {"name": "ref1", "half_width": 0.25,  "step": 1e-2, "time_limit": 5.0},
    {"name": "ref2", "half_width": 0.06,  "step": 2e-3, "time_limit": 5.0},
    {"name": "ref3", "half_width": 0.015, "step": 5e-4, "time_limit": 5.0},
    {"name": "ref4", "half_width": 0.004, "step": 1e-4, "time_limit": 5.0},
    {"name": "ref5", "half_width": 0.001, "step": 1e-5, "time_limit": 5.0},
]


# -----------------------------
# Структуры
# -----------------------------

@dataclass(frozen=True)
class EvalKey:
    temp_s: str
    time_limit: float


@dataclass
class EvalResult:
    temp: float
    temp_s: str
    time_limit: float
    ok: bool
    steps: int
    unperf: float
    dt_sec: float
    raw: dict


# -----------------------------
# Утилиты
# -----------------------------

TEMP_DECIMALS = 9


def norm_temp(x: float) -> float:
    # Округляем всегда до 9 знаков после запятой [web:332]
    return round(float(x), TEMP_DECIMALS)


def temp_to_str(x: float) -> str:
    # Всегда печатаем/передаем ровно 9 знаков после запятой [web:332][web:278]
    return f"{norm_temp(x):.{TEMP_DECIMALS}f}"


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [norm_temp(a)]
    step = (b - a) / (n - 1)
    return [norm_temp(a + i * step) for i in range(n)]


def arange_center(center: float, half_width: float, step: float, lo: float, hi: float) -> List[float]:
    c = norm_temp(center)
    if step <= 0:
        return [norm_temp(clamp(c, lo, hi))]

    m = int(round(half_width / step))
    out = [norm_temp(clamp(c + k * step, lo, hi)) for k in range(-m, m + 1)]

    # uniq по строковому представлению (после округления до 9 знаков)
    seen = set()
    uniq = []
    for t in out:
        ts = temp_to_str(t)
        if ts not in seen:
            seen.add(ts)
            uniq.append(norm_temp(float(ts)))
    return uniq


def is_solved(r: EvalResult) -> bool:
    return r.ok and r.unperf == 0.0


def better_by_unperf(a: EvalResult, b: EvalResult) -> bool:
    """
    True если a лучше b.
    Приоритет:
    1) ok=True лучше ok=False
    2) меньший unperf лучше
    3) меньший steps лучше
    4) меньший dt_sec лучше
    """
    if a.ok != b.ok:
        return a.ok and (not b.ok)
    if a.unperf != b.unperf:
        return a.unperf < b.unperf
    if a.steps != b.steps:
        return a.steps < b.steps
    return a.dt_sec < b.dt_sec


# -----------------------------
# Запуск astar.py
# -----------------------------

def run_astar_once(
    astar_path: Path,
    temp: float,
    time_limit: float,
    python_exe: str,
    data_path: str,
    timeout_pad: float = 1.0,
) -> EvalResult:
    t = norm_temp(temp)
    t_s = temp_to_str(t)
    tl = float(time_limit)

    cmd = [
        python_exe,
        str(astar_path),
        f"--temp={t_s}",            # уже округлено до 9 знаков [web:332]
        f"--time_limit={tl}",
        "--runs=1",
        "--jsonl=",
        f"--data_path={data_path}",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # timeout=tl + timeout_pad,
            env=env,
        )
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return EvalResult(
            temp=t, temp_s=t_s, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "timeout"}
        )

    dt = time.perf_counter() - t0

    if p.returncode != 0:
        return EvalResult(
            temp=t, temp_s=t_s, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "returncode", "stderr": p.stderr[-4000:], "stdout": p.stdout[-4000:]}
        )

    lines = p.stdout.strip().splitlines()
    if not lines:
        return EvalResult(
            temp=t, temp_s=t_s, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "no_stdout"}
        )

    try:
        d = json.loads(lines[-1])
        steps = int(d.get("steps", 10**12))
        unperf = float(d.get("unperf", 10**18))
        dt_sec = float(d.get("dt_sec", dt))

        tt = norm_temp(float(d.get("temp", t)))
        return EvalResult(
            temp=tt,
            temp_s=temp_to_str(tt),
            time_limit=tl,
            ok=True,
            steps=steps,
            unperf=unperf,
            dt_sec=dt_sec,
            raw=d
        )
    except Exception:
        return EvalResult(
            temp=t, temp_s=t_s, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "bad_json", "stdout": p.stdout[-4000:]}
        )


def eval_many(
    astar_path: Path,
    temps: List[float],
    time_limit: float,
    python_exe: str,
    workers: int,
    cache: Dict[EvalKey, EvalResult],
    data_path: str,
    log_path: Optional[Path] = None,
) -> List[EvalResult]:
    tl = float(time_limit)

    # uniq по строке temp (после округления до 9 знаков)
    seen = set()
    uniq = []
    for t in temps:
        t = norm_temp(t)
        t_s = temp_to_str(t)
        if t_s not in seen:
            seen.add(t_s)
            uniq.append(t)

    results: List[EvalResult] = []
    futs = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t in uniq:
            key = EvalKey(temp_to_str(t), tl)
            if key in cache:
                results.append(cache[key])
                continue
            futs.append(ex.submit(run_astar_once, astar_path, t, tl, python_exe, data_path))

        for fut in as_completed(futs):
            r = fut.result()
            cache[EvalKey(r.temp_s, tl)] = r
            results.append(r)

            if log_path:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "temp": r.temp_s,
                        "time_limit": r.time_limit,
                        "ok": r.ok,
                        "steps": r.steps,
                        "unperf": r.unperf,
                        "dt_sec": r.dt_sec,
                        "ts": time.time(),
                    }, ensure_ascii=False) + "\n")

    return results


# -----------------------------
# Итеративное уточнение
# -----------------------------

def pick_best_by_unperf(results: List[EvalResult]) -> EvalResult:
    best = results[0]
    for r in results[1:]:
        if better_by_unperf(r, best):
            best = r
    return best


def tune_temp(
    astar_path: Path,
    t_min: float,
    t_max: float,
    workers: int,
    python_exe: str,
    log_path: Optional[Path],
    data_path: str,
) -> EvalResult:
    cache: Dict[EvalKey, EvalResult] = {}

    init_cfg = REFINE_SCHEDULE[0]
    init_n = int(init_cfg["grid_n"])
    budget = float(init_cfg["time_limit"])

    print(f"[Init] grid n={init_n} budget={budget}s range=[{t_min},{t_max}]")
    temps0 = linspace(t_min, t_max, init_n)

    r0 = eval_many(
        astar_path, temps0,
        time_limit=budget,
        python_exe=python_exe,
        workers=workers,
        cache=cache,
        data_path=data_path,
        log_path=log_path
    )
    best = pick_best_by_unperf(r0)
    print(f"  best0: temp={best.temp_s} ok={best.ok} unperf={best.unperf:.6g} steps={best.steps} dt={best.dt_sec:.4f}")

    for i, cfg in enumerate(REFINE_SCHEDULE[1:], 1):
        half_w = float(cfg["half_width"])
        step = float(cfg["step"])
        budget = float(cfg["time_limit"])

        temps = arange_center(best.temp, half_width=half_w, step=step, lo=t_min, hi=t_max)

        if len(temps) > 600:
            idxs = [int(j * (len(temps) - 1) / (600 - 1)) for j in range(600)]
            temps = [temps[j] for j in idxs]

        rr = eval_many(
            astar_path, temps,
            time_limit=budget,
            python_exe=python_exe,
            workers=workers,
            cache=cache,
            data_path=data_path,
            log_path=log_path
        )
        best = pick_best_by_unperf(rr)

        print(
            f"[Ref {i}] name={cfg['name']} budget={budget}s "
            f"center={best.temp_s} unperf={best.unperf:.6g} steps={best.steps} ok={best.ok} "
            f"(half_width={half_w}, step={step}, points={len(temps)})"
        )

    return best


# -----------------------------
# __main__
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--astar_path", type=str, default="GroupProblem/True_Python/src/astar.py")
    parser.add_argument("--data_path", type=str, default="GroupProblem/True_Python/data/outputs/BIRDS_7/parsed_data.json")

    # Пределы по temp: 1..5
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=5.0)

    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--log_path", type=str, default="tune_log.jsonl")
    parser.add_argument("--out_json", type=str, default="chosen_temp.json")
    args = parser.parse_args()

    ASTAR_PATH = Path(args.astar_path).resolve()
    DATA_PATH = Path(args.data_path).resolve()
    LOG_PATH = Path(args.log_path).resolve()
    OUT_JSON = Path(args.out_json).resolve()

    if not ASTAR_PATH.exists():
        print(f"ERROR: astar.py not found: {ASTAR_PATH}", file=sys.stderr)
        raise SystemExit(2)

    if not DATA_PATH.exists():
        print(f"ERROR: parsed_data.json not found: {DATA_PATH}", file=sys.stderr)
        raise SystemExit(2)

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", encoding="utf-8"):
            pass
    except Exception as e:
        print(f"WARNING: cannot reset log file {LOG_PATH}: {e}", file=sys.stderr)

    print("=== TEMP TUNER START ===")
    print(f"astar: {ASTAR_PATH}")
    print(f"data: {DATA_PATH}")
    print(f"range: [{args.t_min}, {args.t_max}]")
    print(f"workers: {args.workers}")
    print(f"log: {LOG_PATH}")
    print(f"out: {OUT_JSON}")

    t0 = time.perf_counter()
    best = tune_temp(
        astar_path=ASTAR_PATH,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        workers=int(args.workers),
        python_exe=sys.executable,
        log_path=LOG_PATH,
        data_path=str(DATA_PATH),
    )
    total = time.perf_counter() - t0

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "temp": best.temp_s,             # ровно 9 знаков после запятой [web:332]
        "solved": is_solved(best),
        "steps": best.steps,
        "unperf": best.unperf,
        "dt_sec": best.dt_sec,
        "time_limit": best.time_limit,
        "total_tune_sec": total,
        "data_path": str(DATA_PATH),
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(result_payload, ensure_ascii=False))
    print("=== TEMP TUNER END ===")
