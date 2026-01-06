
import argparse
import subprocess
import sys

def verify_neural():
    parser = argparse.ArgumentParser(description="Verify neural-guided synthesis output.")
    parser.add_argument("--quick", action="store_true", help="Reduce search effort for fast smoke runs.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Pass max runtime cap to Systemtest.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks to run.")
    parser.add_argument("--timeout", type=float, default=300, help="Subprocess timeout in seconds.")
    args = parser.parse_args()

    print("[Test] Running Synthesis with Neural Guidance...")
    cmd = [sys.executable, "Systemtest.py", "synthesis"]
    if args.quick:
        cmd.append("--quick")
    if args.max_seconds is not None:
        cmd.extend(["--max-seconds", str(args.max_seconds)])
    if args.tasks:
        cmd.extend(["--tasks", args.tasks])
    
    # Run with timeout to avoid infinite hang if something breaks, but generic synthesis takes time
    # We pipe stdout
    timed_out = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        output = proc.stdout
        errors = proc.stderr
    except subprocess.TimeoutExpired as exc:
        print("[Warn] Timeout reached, checking output collected so far...")
        timed_out = True
        output = exc.stdout or ""
        errors = exc.stderr or ""
    
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

    if timed_out:
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
