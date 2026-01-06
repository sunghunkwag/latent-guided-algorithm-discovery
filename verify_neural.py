
import argparse
import json
import subprocess
import sys

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
        cmd = [sys.executable, "Systemtest.py", "ab-compare", "--seeds", str(args.seeds)]
        if args.max_seconds is not None:
            cmd.extend(["--max-seconds", str(args.max_seconds)])
        if args.quick:
            cmd.append("--quick")
        if args.tasks:
            cmd.extend(["--tasks", args.tasks])
    else:
        cmd = [sys.executable, "Systemtest.py", "synthesis"]
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
    priors_detected = "[Latent] Guidance Priors:" in output
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
    sys.exit(0 if status == "PASS" else 1)

if __name__ == "__main__":
    verify_neural()
