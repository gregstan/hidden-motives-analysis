# Notes for AI Coding Agents

## create_new_file

The global `create_new_file` flag in `general_settings` (config.py) should almost **never** be set to `True`.

Setting it globally to `True` tells every caching function in the codebase to regenerate its output from scratch, overwriting whatever is already on disk. Because most computations are expensive (Bayesian fitting, IC analysis, AMPD matrix, architecture compression, etc.), this risks wiping out hours or days of correct results and forcing a full re-run.

The correct pattern is to pass `create_new_file=True` as a **per-call argument** to the specific function whose output you actually want to refresh. Every caching function in this codebase accepts this argument. Surgical, targeted invalidation is always preferable to a global overwrite.

The global flag exists only for convenience during a fresh environment setup where *nothing* exists yet. In all other situations, leave it `False` in config.py and pass it explicitly at the call site when needed.
