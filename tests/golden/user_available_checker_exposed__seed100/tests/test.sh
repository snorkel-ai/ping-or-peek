#!/bin/bash
# Hidden verifier: uploaded to /tests only after the agent phase ends.
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/verify.py --expected /tests/expected.json --workspace /workspace --out /logs/verifier
