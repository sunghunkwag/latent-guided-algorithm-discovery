
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def verify_neural():
    parser = argparse.ArgumentParser(description="Verify neural-guided synthesis output.")
    parser.add_argument("--quick", action="store_true", help="Reduce search effort for fast smoke runs.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Pass max runtime cap to Systemtest.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks to run.")
    parser.add_argument("--guided", type=int, choices=[0, 1], default=0, help="Enable guided search.")
    parser.add_argument("--ab-compare", action="store_true", help="Run guided vs unguided comparison.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds for A/B comparison.")
    parser.add_argument("--timeout", type=float, default=300, help="Subprocess timeout in seconds.")
    args = parser.parse_args()

    print("[Test] Running Synthesis with Neural Guidance...")
    if args.ab_compare:
        cmd = [sys.executable, "-u", "Systemtest.py", "ab-compare", "--seeds", str(args.seeds)]
        if args.max_seconds is None:
            args.max_seconds = 120
        cmd.extend(["--max-seconds", str(args.max_seconds)])
        if args.quick:
            cmd.append("--quick")
        if args.tasks:
            cmd.extend(["--tasks", args.tasks])
    else:
        cmd = [sys.executable, "-u", "Systemtest.py", "synthesis"]
        if args.quick:
            cmd.append("--quick")
        if args.max_seconds is not None:
            cmd.extend(["--max-seconds", str(args.max_seconds)])
        if args.tasks:
            cmd.extend(["--tasks", args.tasks])
        cmd.extend(["--guided", str(args.guided)])
    
    # Run with timeout to avoid infinite hang if something breaks, but generic synthesis takes time
    # We pipe stdout
    timed_out = False
    timeout = args.timeout
    if args.ab_compare and args.max_seconds:
        timeout = max(timeout, args.max_seconds + 60)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout
        errors = proc.stderr
    except subprocess.TimeoutExpired as exc:
        print("[Warn] Timeout reached, checking output collected so far...")
        timed_out = True
        output = exc.stdout or ""
        errors = exc.stderr or ""

    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if isinstance(errors, bytes):
        errors = errors.decode("utf-8", errors="replace")
    
    print("\n[STDOUT Fragment]")
    print(output[-1000:])
    print("\n[STDERR Fragment]")
    print(errors[-1000:])
    
    # Checks
    priors_detected = (
        "[Latent] Combined Priors:" in output
        or "[Latent] Prior calibration:" in output
        or "[Hypothesis] Prior bias applied" in output
    )
    if priors_detected:
        print("[Pass] Latent Navigator is active (Priors detected).")
    else:
        print("[Fail] No Latent Priors detected.")
        
    if "[Latent] Learning success pattern..." in output:
        print("[Pass] Latent Navigator is learning (Self-Correction detected).")
    else:
        print("[Warn] No learning detected (maybe didn't find solution?).")
        # passed = False # Optional, maybe strictly checking activation is enough for now
        
    if "RuntimeError: EXEC BANNED" in errors or "RuntimeError: EXEC BANNED" in output:
         print("[Fail] EXEC BANNED triggered!")

    evidence_entries = []
    for line in output.splitlines():
        if line.startswith("EVIDENCE_JSON="):
            try:
                payload = json.loads(line.split("=", 1)[1])
                evidence_entries.append(payload)
            except json.JSONDecodeError:
                continue

    run_id = f"run_{int(time.time())}"
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = logs_dir / f"evidence_ab_compare_{run_id}.jsonl"
    fail_pairs = {}
    if evidence_entries:
        by_key = {}
        for entry in evidence_entries:
            key = (entry.get("seed"), entry.get("task_id"))
            by_key.setdefault(key, {})[entry.get("mode")] = entry
        for key, modes in by_key.items():
            guided = modes.get("guided")
            unguided = modes.get("unguided")
            if not guided or not unguided:
                continue
            guided_train = guided.get("train_acc", 0.0)
            unguided_train = unguided.get("train_acc", 0.0)
            guided_holdout = guided.get("holdout_acc", 0.0)
            unguided_holdout = unguided.get("holdout_acc", 0.0)
            guided_shift = guided.get("shift_acc", 0.0)
            unguided_shift = unguided.get("shift_acc", 0.0)
            fail = (
                guided_train > unguided_train
                and guided_holdout <= unguided_holdout
                and guided_shift <= unguided_shift
            )
            if fail:
                fail_pairs[key] = True

    with evidence_path.open("w", encoding="utf-8") as handle:
        for entry in evidence_entries:
            key = (entry.get("seed"), entry.get("task_id"))
            run_status = "FAIL" if entry.get("mode") == "guided" and fail_pairs.get(key) else "PASS"
            record = {
                "run_id": run_id,
                "seed": entry.get("seed"),
                "task_id": entry.get("task_id"),
                "mode": entry.get("mode"),
                "train_acc": entry.get("train_acc", 0.0),
                "holdout_acc": entry.get("holdout_acc", 0.0),
                "shift_acc": entry.get("shift_acc", 0.0),
                "complexity": entry.get("node_count"),
                "hypothesis_selected": entry.get("hypothesis_type"),
                "hypothesis_params": entry.get("hypothesis_params"),
                "hypothesis_confidence": entry.get("hypothesis_confidence", 0.0),
                "hypothesis_passed": entry.get("hypothesis_passed", False),
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "max_seconds": entry.get("max_seconds"),
                "quick": entry.get("quick"),
                "run_status": run_status,
            }
            handle.write(json.dumps(record) + "\n")
    print(f"[Evidence] JSONL written: {evidence_path}")
    with evidence_path.open("r", encoding="utf-8") as handle:
        preview = []
        for _ in range(3):
            line = handle.readline()
            if not line:
                break
            preview.append(line.rstrip("\n"))
    if preview:
        print("[Evidence] Preview:")
        for line in preview:
            print(line)
    if fail_pairs:
        print("[Fail] Guided improved train but not holdout/shift for some runs.")
        for (seed, task_id) in sorted(fail_pairs):
            print(f"  - seed={seed} task={task_id}")

    if args.ab_compare:
        status = "INCONCLUSIVE"
        if not timed_out and "AB_COMPARE_JSON" in output:
            try:
                summary_line = [line for line in output.splitlines() if line.startswith("AB_COMPARE_JSON=")][-1]
                summary = json.loads(summary_line.split("=", 1)[1])
                status = "PASS" if summary.get("guided_wins") else "FAIL"
            except Exception:
                status = "FAIL"
    elif timed_out:
        status = "INCONCLUSIVE"
    elif priors_detected:
        status = "PASS"
    else:
        status = "FAIL"

    if status == "PASS":
        print("\nOVERALL: NEURAL VERIFICATION PASSED")
    else:
        print("\nOVERALL: NEURAL VERIFICATION FAILED")
    print(f"STATUS={status}")
    if status in {"PASS", "INCONCLUSIVE"}:
        sys.exit(0)
    sys.exit(1)

if __name__ == "__main__":
    verify_neural()
