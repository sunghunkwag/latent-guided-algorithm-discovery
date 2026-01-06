# Latent-Guided Algorithm Discovery (Local Verification)

This repository contains a local verification harness for a lightweight algorithm discovery system. The focus is on deterministic, safe execution of synthesized recursive programs and on exposing intermediate signals (such as latent guidance priors) during synthesis so runs can be inspected and validated on a single machine.

## What this repo is NOT
This codebase is not a trained deep learning model, and it does not claim benchmark performance. The verification suite is a correctness and observability check, not a competitive evaluation.

## Quick start

```bash
python verify_neural.py --quick --max-seconds 45
python -c "import Systemtest; Systemtest.run_synthesis_verification_suite()"
python verify_neural.py --ab-compare --seeds 5 --max-seconds 600
```

## Interpreting PASS / FAIL / INCONCLUSIVE

* **PASS**: The run completed and latent priors were detected (e.g., output contains `"[Latent] Guidance Priors:"`).
* **FAIL**: The run completed but latent priors were not detected.
* **INCONCLUSIVE**: The run exceeded the timeout. Partial output may exist, but the run did not finish.
