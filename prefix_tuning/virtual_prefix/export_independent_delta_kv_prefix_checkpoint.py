#!/usr/bin/env python3
"""Backward-compatible entry point for Independent Delta Prefix checkpoints."""

try:
    from prefix_tuning.virtual_prefix.merge_prefix_checkpoint import main
except ModuleNotFoundError:  # Direct script execution before editable install.
    from merge_prefix_checkpoint import main


if __name__ == "__main__":
    main(expected_trainable_type="independent_delta_kv_prefix")
