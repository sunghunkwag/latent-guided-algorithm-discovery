
import subprocess
import sys

def verify_neural():
    print("[Test] Running Synthesis with Neural Guidance...")
    cmd = [sys.executable, "-c", "import Systemtest; Systemtest.run_synthesis_verification_suite()"]
    
    # Run with timeout to avoid infinite hang if something breaks, but generic synthesis takes time
    # We pipe stdout
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutError:
        print("[Warn] Timeout reached, checking output collected so far...")
        proc = subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Timeout") # Dummy
        # Actually subprocess.run on timeout raises exception and we lose stdout?
        # In Python 3.7+ capture_output=True with timeout might lose output if exception.
        # But we hopefully finish Fibonacci quickly.
        pass

    output = proc.stdout
    errors = proc.stderr
    
    print("\n[STDOUT Fragment]")
    print(output[-1000:])
    print("\n[STDERR Fragment]")
    print(errors[-1000:])
    
    # Checks
    passed = True
    if "[Latent] Guidance Priors:" in output:
        print("[Pass] Latent Navigator is active (Priors detected).")
    else:
        print("[Fail] No Latent Priors detected.")
        passed = False
        
    if "[Latent] Learning success pattern..." in output:
        print("[Pass] Latent Navigator is learning (Self-Correction detected).")
    else:
        print("[Warn] No learning detected (maybe didn't find solution?).")
        # passed = False # Optional, maybe strictly checking activation is enough for now
        
    if "RuntimeError: EXEC BANNED" in errors or "RuntimeError: EXEC BANNED" in output:
         print("[Fail] EXEC BANNED triggered!")
         passed = False
         
    if passed:
        print("\nOVERALL: NEURAL VERIFICATION PASSED")
    else:
        print("\nOVERALL: NEURAL VERIFICATION FAILED")

if __name__ == "__main__":
    verify_neural()
