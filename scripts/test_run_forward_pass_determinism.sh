#!/usr/bin/env bash
set -euo pipefail

cargo test run_forward_pass_is_deterministic_for_fixed_input -- --nocapture
