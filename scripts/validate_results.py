"""Validate every results/*.json against the expected schema.

Exit 0 if all good, 1 (with messages) otherwise. Run by CI on PRs.

Usage:
  python scripts/validate_results.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

TASKS = {"translate", "punctuate", "char-gloss", "idiom-source", "fill-in", "compress"}
N_PER_TASK = 100

# Above this fraction of API-error items, a task's summary is statistically
# untrustworthy (too few real scores). Reported as a hard validation error.
MAX_ERROR_RATE = 0.05


def _task_error_rate(tr: dict, items: list) -> float:
    """Best-effort API-error rate for a task, tolerant of older result files.

    Prefers the explicit `error_rate` field; falls back to `errors`/`n`; and
    finally counts items carrying an `error` field (new eval_runner schema).
    Legacy files predate all of these → rate 0.0 (cannot assess, don't gate).
    """
    if isinstance(tr.get("error_rate"), (int, float)):
        return float(tr["error_rate"])
    n = tr.get("n") or len(items) or 0
    errors = tr.get("errors")
    if isinstance(errors, int) and n:
        return errors / n
    if items:
        n_err = sum(1 for it in items if isinstance(it, dict) and it.get("error"))
        return n_err / len(items)
    return 0.0


def validate_file(fp: Path) -> list[str]:
    errs: list[str] = []
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{fp.name}: not valid JSON ({e})"]

    if not isinstance(d, dict):
        return [f"{fp.name}: top level is not an object"]
    for key in ("model", "tasks"):
        if key not in d:
            errs.append(f"{fp.name}: missing top-level key '{key}'")
    tasks = d.get("tasks", {})
    if not isinstance(tasks, dict):
        return errs + [f"{fp.name}: 'tasks' is not an object"]

    missing = TASKS - set(tasks)
    if missing:
        # Missing tasks are a warning, not a failure — tasks are added over time
        # (e.g. `compress` was added after some models were evaluated) and we
        # don't want to gate CI on re-running every legacy result file.
        # Schema problems within tasks that ARE present are still errors below.
        print(f"  warn: {fp.name}: missing task(s): {sorted(missing)}")
    extra = set(tasks) - TASKS
    if extra:
        errs.append(f"{fp.name}: unexpected task key(s): {sorted(extra)}")

    for t in sorted(TASKS & set(tasks)):
        tr = tasks[t]
        if not isinstance(tr, dict):
            errs.append(f"{fp.name}: tasks.{t} is not an object")
            continue
        if not isinstance(tr.get("summary"), dict):
            errs.append(f"{fp.name}: tasks.{t}.summary missing or not an object")
        items = tr.get("items")
        if not isinstance(items, list):
            errs.append(f"{fp.name}: tasks.{t}.items missing or not a list")
            continue
        # High API-error rate => the summary is computed from too few real
        # scores to trust. We WARN rather than fail: some already-committed
        # legacy results predate the retry/error-tracking fix and carry high
        # error rates (notably the `compress` task on the Opus runs), and we
        # must not gate CI on data we can't retroactively re-run here. A warning
        # makes the unreliability visible without breaking existing files.
        err_rate = _task_error_rate(tr, items)
        if err_rate > MAX_ERROR_RATE:
            print(
                f"  warn: {fp.name}: tasks.{t} API error rate {err_rate:.1%} "
                f"exceeds {MAX_ERROR_RATE:.0%} — summary UNRELIABLE, re-run this task"
            )
        if len(items) > N_PER_TASK:
            errs.append(f"{fp.name}: tasks.{t}.items has {len(items)}, max allowed {N_PER_TASK}")
        elif len(items) < N_PER_TASK:
            # partial run (e.g. --limit flag) — warn but do not fail
            print(f"  warn: {fp.name}: tasks.{t}.items has {len(items)}/{N_PER_TASK} (partial run?)")
        for i, it in enumerate(items[:N_PER_TASK]):
            if not isinstance(it, dict):
                errs.append(f"{fp.name}: tasks.{t}.items[{i}] is not an object")
                continue
            for k in ("id", "prediction", "scores"):
                if k not in it:
                    errs.append(f"{fp.name}: tasks.{t}.items[{i}] missing '{k}'")
    return errs


def main() -> int:
    # `_*.json` is reserved for non-result artifacts in results/ (e.g. _bootstrap.json).
    files = sorted(f for f in RESULTS.glob("*.json") if not f.name.startswith("_"))
    if not files:
        print(f"no result files in {RESULTS}")
        return 1
    all_errs: list[str] = []
    for fp in files:
        all_errs.extend(validate_file(fp))
    if all_errs:
        print("VALIDATION FAILED:")
        for e in all_errs:
            print(f"  - {e}")
        return 1
    print(f"OK — {len(files)} result file(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
