"""Launch pinned tau2 with explicit judge configuration; keep official scoring."""
import json
import os
import tau2.config as config

if os.getenv("TAU2_LLM_NL_ASSERTIONS"):
    config.DEFAULT_LLM_NL_ASSERTIONS = os.environ["TAU2_LLM_NL_ASSERTIONS"]
if os.getenv("TAU2_LLM_NL_ASSERTIONS_ARGS"):
    config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = json.loads(os.environ["TAU2_LLM_NL_ASSERTIONS_ARGS"])

from tau2.cli import main
if __name__ == "__main__":
    main()
