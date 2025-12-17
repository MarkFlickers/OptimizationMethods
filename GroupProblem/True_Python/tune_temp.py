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


def is_solved(r: EvalResult) -> bool:
    return r.ok and r.unperf == 0.0


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
    # уникализация с сохранением порядка
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# -----------------------------
# Запуск astar.py
# -----------------------------

def run_astar_once(
    astar_path: Path,
    temp: float,
    time_limit: float,
    python_exe: str,
    timeout_pad: float = 0.5,
) -> EvalResult:
    t = round5(temp)
    tl = float(time_limit)

    cmd = [
        python_exe,
        str(astar_path),
        f"--temp={t:.5f}",
        f"--time_limit={tl}",
        "--runs=1",
        "--jsonl=",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=tl + timeout_pad,
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
    log_path: Optional[Path] = None,
) -> List[EvalResult]:
    tl = float(time_limit)

    # uniq temps (до 5 знаков)
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
            futs.append(ex.submit(run_astar_once, astar_path, t, tl, python_exe))

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
# Логика “перспективности”
# -----------------------------

def border_promise(r: EvalResult) -> float:
    """
    Чем МЕНЬШЕ, тем ЛУЧШЕ.

    Идея:
    - Решение (unperf==0) всегда топ (просто по steps).
    - Если не решено:
      * если dt ~ time_limit => это “живой поиск”, вокруг такого temp есть смысл копать.
      * если dt << time_limit => вероятно поиск быстро схлопнулся/тупик => плохо.
      * unperf важен, но не единственный: dt/limit тоже сигнал.

    Возвращаем “оценку” для отбора кандидатов на следующий раунд.
    """
    if not r.ok:
        return 1e30

    if is_solved(r):
        # Решил => лучше по steps, потом по времени
        return r.steps + 0.01 * r.dt_sec

    # Не решил:
    # коэффициент использования бюджета (0..1+)
    use = r.dt_sec / max(r.time_limit, 1e-9)

    # если закончил слишком быстро, но не решил — жестко штрафуем
    # (обычно такие точки бесполезны для локального поиска)
    early_penalty = 0.0
    if use < 0.65:
        early_penalty = 1e9 * (0.65 - use)

    # unperf нормируем “мягко” (log не используем, чтобы не ловить отриц./нулевые)
    # просто оставим как есть, но добавим бонус “за борьбу” (близость к лимиту)
    fight_bonus = (1.0 - min(use, 1.0)) * 1000.0  # чем ближе к лимиту, тем меньше штраф

    return early_penalty + r.unperf + fight_bonus


def pick_top_diverse_by_metric(
    results: List[EvalResult],
    k: int,
    min_dist: float,
    metric_fn,
) -> List[EvalResult]:
    """
    Сортируем по metric_fn, берём top-k с разнесением по temp.
    """
    ordered = sorted(results, key=metric_fn)
    picked: List[EvalResult] = []
    for r in ordered:
        if len(picked) >= k:
            break
        if all(abs(r.temp - p.temp) >= min_dist for p in picked):
            picked.append(r)
    return picked if picked else ordered[:k]


def best_solved(results: List[EvalResult]) -> Optional[EvalResult]:
    solved = [r for r in results if is_solved(r)]
    if not solved:
        return None
    return min(solved, key=lambda r: (r.steps, r.dt_sec, r.temp))


# -----------------------------
# Основной тюнинг
# -----------------------------

def tune_temp(
    astar_path: Path,
    t_min: float,
    t_max: float,
    workers: int,
    python_exe: str,
    log_path: Optional[Path],
) -> EvalResult:
    cache: Dict[EvalKey, EvalResult] = {}

    print("\n[Phase 0] Coarse scan @0.10s")
    coarse = linspace(t_min, t_max, 180)  # достаточно плотная сетка
    r0 = eval_many(astar_path, coarse, time_limit=0.10, python_exe=python_exe,
                   workers=workers, cache=cache, log_path=log_path)

    solved0 = best_solved(r0)
    if solved0:
        print(f"  solved found already: temp={solved0.temp:.5f} steps={solved0.steps} dt={solved0.dt_sec:.4f}")
    else:
        print("  no solved on 0.10s budget")

    # Берём:
    # - лучшие решившие (если есть)
    # - + “перспективные границы” по border_promise
    top_border0 = pick_top_diverse_by_metric(r0, k=30, min_dist=0.03, metric_fn=border_promise)

    print("  top candidates (0.10s) by promise:")
    for r in top_border0[:10]:
        print(f"    temp={r.temp:.5f} solved={is_solved(r)} steps={r.steps} unperf={r.unperf:.3f} dt={r.dt_sec:.4f}")

    print("\n[Phase 1] Promote candidates @0.50s")
    temps1 = [r.temp for r in top_border0]
    r1 = eval_many(astar_path, temps1, time_limit=0.50, python_exe=python_exe,
                   workers=workers, cache=cache, log_path=log_path)

    solved1 = best_solved(r1)
    if solved1:
        print(f"  best solved @0.50s: temp={solved1.temp:.5f} steps={solved1.steps} dt={solved1.dt_sec:.4f}")
    else:
        print("  no solved on 0.50s budget")

    # Вокруг самых “границ” строим локальные окна (как у тебя: рядом может внезапно решать)
    seeds1 = pick_top_diverse_by_metric(r1, k=8, min_dist=0.02, metric_fn=border_promise)

    print("\n[Phase 2] Local exploration around promising seeds @1.50s")
    temps2: List[float] = []
    for s in seeds1:
        # окно ±0.15 с шагом 0.01 — быстро находит “карманы” решаемости
        temps2 += neighborhood(s.temp, half_width=0.15, step=0.01, lo=t_min, hi=t_max)
    r2 = eval_many(astar_path, temps2, time_limit=1.50, python_exe=python_exe,
                   workers=workers, cache=cache, log_path=log_path)

    solved2 = best_solved(r2)
    if solved2:
        print(f"  best solved @1.50s: temp={solved2.temp:.5f} steps={solved2.steps} dt={solved2.dt_sec:.4f}")
    else:
        print("  still no solved on 1.50s budget (will continue anyway)")

    # Определяем текущий “центр”:
    # если есть решённые — центр по лучшему steps
    # иначе — центр по лучшему promise
    if solved2:
        center = solved2.temp
        print(f"\n[Phase 3] Refinement around best solved center={center:.5f} (focus: min steps)")
    else:
        best_uns = min(r2, key=border_promise)
        center = best_uns.temp
        print(f"\n[Phase 3] Refinement around best UNSOLVED center={center:.5f} (focus: find any solved)")

    # Финальное уточнение: уменьшаем шаг до 5 знаков.
    # Важно: на финале даём чуть больше времени, иначе “ложно нерешаемо”.
    refinement = [
        (0.050, 0.002, 2.0),
        (0.010, 0.0005, 2.0),
        (0.002, 0.0001, 2.0),
        (0.0005, 0.00002, 2.0),  # даёт точность до 5 знаков
    ]

    best_global: Optional[EvalResult] = None

    for i, (half_w, step, budget) in enumerate(refinement, 1):
        temps = neighborhood(center, half_width=half_w, step=step, lo=t_min, hi=t_max)
        rr = eval_many(astar_path, temps, time_limit=budget, python_exe=python_exe,
                       workers=workers, cache=cache, log_path=log_path)

        solved_rr = best_solved(rr)
        if solved_rr:
            candidate = solved_rr
            # если нашли решённое — дальше центрируемся по нему
            center = candidate.temp
            print(f"  [Ref {i}] solved: temp={candidate.temp:.5f} steps={candidate.steps} dt={candidate.dt_sec:.4f} (budget={budget})")
        else:
            # если всё ещё нет решённого — продолжаем по “границе”
            candidate = min(rr, key=border_promise)
            center = candidate.temp
            print(f"  [Ref {i}] unsolved best-promise: temp={candidate.temp:.5f} unperf={candidate.unperf:.3f} dt={candidate.dt_sec:.4f} (budget={budget})")

        if best_global is None:
            best_global = candidate
        else:
            # сравнение: решённость -> steps -> time -> temp
            def final_key(r: EvalResult):
                return (0 if is_solved(r) else 1, r.steps, r.dt_sec, r.temp)
            if final_key(candidate) < final_key(best_global):
                best_global = candidate

    assert best_global is not None
    return best_global


# -----------------------------
# __main__ (без аргументов)
# -----------------------------

if __name__ == "__main__":
    # "Аргументы" — задаются тут:
    ASTAR_PATH = Path("src/astar.py").resolve()
    T_MIN = 1.0
    T_MAX = 10.0

    # ВАЖНО: subprocess + python = лучше не ставить workers=cpu_count,
    # иначе система может начать троттлить.
    WORKERS = 16

    # Лог тюнера (можно отключить None)
    LOG_PATH = Path("tune_log.jsonl").resolve()

    print("=== TEMP TUNER START ===")
    print(f"astar: {ASTAR_PATH}")
    print(f"range: [{T_MIN}, {T_MAX}]")
    print(f"workers: {WORKERS}")
    print(f"log: {LOG_PATH}")

    if not ASTAR_PATH.exists():
        print(f"ERROR: astar.py not found: {ASTAR_PATH}")
        raise SystemExit(2)

    t0 = time.perf_counter()
    best = tune_temp(
        astar_path=ASTAR_PATH,
        t_min=T_MIN,
        t_max=T_MAX,
        workers=WORKERS,
        python_exe=sys.executable,
        log_path=LOG_PATH,
    )
    total = time.perf_counter() - t0

    print("\n=== FINAL RESULT ===")
    print(json.dumps({
        "temp": round5(best.temp),
        "solved": is_solved(best),
        "steps": best.steps,
        "unperf": best.unperf,
        "dt_sec": best.dt_sec,
        "time_limit": best.time_limit,
        "total_tune_sec": total,
    }, ensure_ascii=False, indent=2))

    print("\n=== TEMP TUNER END ===")
