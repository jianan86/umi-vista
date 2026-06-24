#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_expected(path: Path | None) -> list[str]:
    if path is None:
        return []
    tasks: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.split("#", 1)[0].strip()
            if item:
                tasks.append(item.split()[0])
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    results = []
    for path in sorted(result_root.rglob("_result.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["_path"] = str(path)
        results.append(row)

    latest_by_task = {}
    duplicates = defaultdict(list)
    for row in results:
        task = row.get("task", "")
        duplicates[task].append(row)
        old = latest_by_task.get(task)
        if old is None or str(row.get("timestamp", "")) >= str(old.get("timestamp", "")):
            latest_by_task[task] = row

    expected = load_expected(args.expected_tasks)
    expected_set = set(expected)
    observed_set = set(latest_by_task)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set) if expected_set else []

    ordered_tasks = expected if expected else sorted(latest_by_task)
    summary_rows = []
    for task in ordered_tasks:
        row = latest_by_task.get(task)
        if row is None:
            summary_rows.append(
                {
                    "task": task,
                    "successes": "",
                    "test_num": "",
                    "success_rate": "",
                    "status": "missing",
                    "skip_get_obs_within_replan": "",
                    "instruction_source": "",
                    "instruction_type": "",
                    "result_path": "",
                }
            )
            continue
        summary_rows.append(
            {
                "task": task,
                "successes": row.get("successes", ""),
                "test_num": row.get("test_num", ""),
                "success_rate": row.get("success_rate", ""),
                "status": "ok",
                "skip_get_obs_within_replan": row.get("skip_get_obs_within_replan", ""),
                "instruction_source": row.get("instruction_source", ""),
                "instruction_type": row.get("instruction_type", ""),
                "result_path": row.get("_path", ""),
            }
        )

    numeric = [
        float(row["success_rate"])
        for row in summary_rows
        if row["status"] == "ok" and row["success_rate"] != ""
    ]
    mean_success_rate = sum(numeric) / len(numeric) if numeric else None

    out_prefix = args.out_prefix or (result_root / "summary")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = out_prefix.with_suffix(".tsv")
    json_path = out_prefix.with_suffix(".json")

    headers = [
        "task",
        "successes",
        "test_num",
        "success_rate",
        "status",
        "skip_get_obs_within_replan",
        "instruction_source",
        "instruction_type",
        "result_path",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in summary_rows:
            f.write("\t".join(str(row.get(key, "")) for key in headers) + "\n")

    payload = {
        "result_root": str(result_root),
        "result_count": len(results),
        "task_count": len(summary_rows),
        "ok_task_count": len(numeric),
        "mean_success_rate": mean_success_rate,
        "missing_tasks": missing,
        "extra_tasks": extra,
        "duplicate_result_counts": {
            task: len(rows) for task, rows in sorted(duplicates.items()) if len(rows) > 1
        },
        "summary_tsv": str(tsv_path),
        "rows": summary_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
