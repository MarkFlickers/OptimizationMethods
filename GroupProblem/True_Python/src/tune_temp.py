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
# Структуры
# -----------------------------

@dataclass(frozen=True)
class EvalKey:
    temp: float
    time_limit: float


@dataclass
class EvalResult:
    temp: float
    time_limit: float
    ok: bool
    steps: int
    unperf: float
    dt_sec: float
    raw: dict


# -----------------------------
# Утилиты
# -----------------------------

def round5(x: float) -> float:
    return float(f"{x:.5f}")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def neighborhood(center: float, half_width: float, step: float, lo: float, hi: float) -> List[float]:
    c = round5(center)
    if step <= 0:
        return [clamp(c, lo, hi)]
    m = int(round(half_width / step))
    out = []
    for k in range(-m, m + 1):
        out.append(round5(clamp(c + k * step, lo, hi)))
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def is_solved(r: EvalResult) -> bool:
    return r.ok and r.unperf == 0.0


def metric_key(r: EvalResult) -> Tuple:
    """
    Чем меньше — тем лучше.
    1) solved всегда выше unsolved
    2) solved: steps, dt
    3) unsolved: unperf, (prefer larger dt) чтобы выбирать "борющиеся" точки
    """
    if not r.ok:
        return (2, 10**18, 10**18, 10**18)
    if is_solved(r):
        return (0, r.steps, r.dt_sec, r.temp)
    return (1, r.unperf, -r.dt_sec, r.temp)


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
    t = round5(temp)
    tl = float(time_limit)

    # DATA = list()
    # with open(data_path, "r", encoding="utf-8") as f:
    #     payload = json.load(f)
    # DATA = payload["DATA"]
    # print("DATA_LEN", len(DATA), "ROW_LEN", len(DATA[0]) if DATA else 0, file=sys.stderr)

    cmd = [
        python_exe,
        str(astar_path),
        f"--temp={t:.5f}",
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
            temp=t, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "timeout"}
        )

    dt = time.perf_counter() - t0

    if p.returncode != 0:
        return EvalResult(
            temp=t, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "returncode", "stderr": p.stderr[-4000:], "stdout": p.stdout[-4000:]}
        )

    lines = p.stdout.strip().splitlines()
    if not lines:
        return EvalResult(
            temp=t, time_limit=tl, ok=False,
            steps=10**12, unperf=10**18, dt_sec=dt,
            raw={"error": "no_stdout"}
        )

    try:
        d = json.loads(lines[-1])
        steps = int(d.get("steps", 10**12))
        unperf = float(d.get("unperf", 10**18))
        dt_sec = float(d.get("dt_sec", dt))
        return EvalResult(
            temp=round5(float(d.get("temp", t))),
            time_limit=tl,
            ok=True,
            steps=steps,
            unperf=unperf,
            dt_sec=dt_sec,
            raw=d
        )
    except Exception:
        return EvalResult(
            temp=t, time_limit=tl, ok=False,
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

    seen = set()
    uniq = []
    for t in temps:
        t5 = round5(t)
        if t5 not in seen:
            seen.add(t5)
            uniq.append(t5)

    results: List[EvalResult] = []
    futs = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t in uniq:
            key = EvalKey(t, tl)
            if key in cache:
                results.append(cache[key])
                continue
            futs.append(ex.submit(run_astar_once, astar_path, t, tl, python_exe, data_path))

        for fut in as_completed(futs):
            r = fut.result()
            cache[EvalKey(round5(r.temp), tl)] = r
            results.append(r)

            if log_path:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "temp": round5(r.temp),
                        "time_limit": r.time_limit,
                        "ok": r.ok,
                        "steps": r.steps,
                        "unperf": r.unperf,
                        "dt_sec": r.dt_sec,
                        "ts": time.time(),
                    }, ensure_ascii=False) + "\n")

    return results


# -----------------------------
# Successive Halving tuner
# -----------------------------

def select_top_diverse(results: List[EvalResult], k: int, min_dist: float) -> List[EvalResult]:
    ordered = sorted(results, key=metric_key)
    picked: List[EvalResult] = []
    for r in ordered:
        if len(picked) >= k:
            break
        if all(abs(r.temp - p.temp) >= min_dist for p in picked):
            picked.append(r)
    return picked if picked else ordered[:k]


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

    init_n = 600
    init_budget = 0.05
    print(f"[Init] scan n={init_n} budget={init_budget}s range=[{t_min},{t_max}]")
    init_temps = linspace(t_min, t_max, init_n)

    r0 = eval_many(
        astar_path, init_temps,
        time_limit=init_budget,
        python_exe=python_exe,
        workers=workers,
        cache=cache,
        data_path=data_path,
        log_path=log_path
    )

    eta = 3
    budgets = [0.10, 0.25, 0.60, 1.20, 2.00]
    candidates = select_top_diverse(r0, k=120, min_dist=0.01)

    print(f"[Rung 0] keep={len(candidates)} (from {len(r0)})")
    best_now = min(candidates, key=metric_key)
    print(f"  best: temp={best_now.temp:.5f} solved={is_solved(best_now)} steps={best_now.steps} unperf={best_now.unperf:.3f} dt={best_now.dt_sec:.4f}")

    rung_results_all: List[EvalResult] = r0[:]

    for rung, bud in enumerate(budgets, 1):
        temps = [c.temp for c in candidates]
        rr = eval_many(
            astar_path, temps,
            time_limit=bud,
            python_exe=python_exe,
            workers=workers,
            cache=cache,
            data_path=data_path,
            log_path=log_path
        )
        rung_results_all += rr

        rr_sorted = sorted(rr, key=metric_key)

        keep = max(10, len(rr_sorted) // eta)
        candidates = select_top_diverse(rr_sorted[: max(keep * 2, keep)], k=keep, min_dist=0.005)

        best_now = rr_sorted[0]
        print(f"[Rung {rung}] budget={bud}s keep={len(candidates)} best: temp={best_now.temp:.5f} solved={is_solved(best_now)} steps={best_now.steps} unperf={best_now.unperf:.3f} dt={best_now.dt_sec:.4f}")

        if any(is_solved(x) for x in rr_sorted[:5]) and len(candidates) <= 12:
            break

    best_global = min(rung_results_all, key=metric_key)
    center = best_global.temp
    print(f"\n[Refine] center={center:.5f} based on best_global solved={is_solved(best_global)} steps={best_global.steps} unperf={best_global.unperf:.3f}")

    refine_plan = [
        (0.20, 0.01, 0.60),
        (0.06, 0.002, 1.20),
        (0.015, 0.0005, 2.00),
        (0.004, 0.0001, 2.00),
    ]

    for i, (half_w, step, bud) in enumerate(refine_plan, 1):
        temps = neighborhood(center, half_width=half_w, step=step, lo=t_min, hi=t_max)

        MAX_TEMPS = 220
        if len(temps) > MAX_TEMPS:
            idxs = [int(j * (len(temps) - 1) / (MAX_TEMPS - 1)) for j in range(MAX_TEMPS)]
            temps = [temps[j] for j in idxs]

        rr = eval_many(
            astar_path, temps,
            time_limit=bud,
            python_exe=python_exe,
            workers=workers,
            cache=cache,
            data_path=data_path,
            log_path=log_path
        )
        best_local = min(rr, key=metric_key)

        if metric_key(best_local) < metric_key(best_global):
            best_global = best_local
            center = best_local.temp

        print(f"  [Ref {i}] budget={bud}s best_local: temp={best_local.temp:.5f} solved={is_solved(best_local)} steps={best_local.steps} unperf={best_local.unperf:.3f} dt={best_local.dt_sec:.4f}")

    return best_global


# -----------------------------
# __main__ (без аргументов)
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--astar_path", type=str, default="GroupProblem/True_Python/src/astar.py")
    parser.add_argument("--data_path", type=str, default="GroupProblem/True_Python/data/outputs/BIRDS_7/parsed_data.json")
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--log_path", type=str, default="tune_log.jsonl")
    parser.add_argument("--out_json", type=str, default="chosen_temp.json")  # куда записать temp
    args = parser.parse_args()

    ASTAR_PATH = Path(args.astar_path).resolve()
    DATA_PATH = Path(args.data_path).resolve()
    LOG_PATH = Path(args.log_path).resolve()
    OUT_JSON = Path(args.out_json).resolve()

    # (1) Проверки
    if not ASTAR_PATH.exists():
        print(f"ERROR: astar.py not found: {ASTAR_PATH}", file=sys.stderr)
        raise SystemExit(2)

    if not DATA_PATH.exists():
        print(f"ERROR: parsed_data.json not found: {DATA_PATH}", file=sys.stderr)
        raise SystemExit(2)

    # (2) Очистка лога (если нужен каждый запуск заново)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", encoding="utf-8") as f:
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

    # (3) Записываем chosen_temp.json (это будет читать e2e)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "temp": round5(best.temp),
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

    # (4) Дополнительно печатаем в stdout, чтобы можно было парсить и без файла
    print(json.dumps(result_payload, ensure_ascii=False))

    print("=== TEMP TUNER END ===")
