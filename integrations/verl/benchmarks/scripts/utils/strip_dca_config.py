#!/usr/bin/env python3
"""Strip `dual_chunk_attention_config` from a HuggingFace checkpoint's config.json.

Why this exists
---------------
Qwen2.5-7B-Instruct-1M (and other Qwen 1M-context checkpoints) ship
`dual_chunk_attention_config` in config.json to enable Dual Chunk Attention (DCA)
for contexts beyond `original_max_position_embeddings` (262144 for that
checkpoint). The vLLM nightly this repo runs against
(0.26.1rc1.dev363+gbeca88e59) has NO dual-chunk attention backend registered at
all - `vllm/v1/attention/backends/registry.py`'s AttentionBackendEnum has no
DUAL_CHUNK_FLASH_ATTN or equivalent - so the config routes into
`FlashAttentionImpl.__init__()` with an unsupported `layer_idx` kwarg and the
engine dies with a TypeError before it ever loads the model. That is a gap in
this specific dev snapshot, not a missing flag on our side: `qwen2.py`/`qwen3.py`
both guard the `layer_idx`/DCA plumbing behind `if dual_chunk_attention_config`,
so removing the key takes the whole path out of play.

Dropping DCA is safe for this repo's benchmarks: they run at 64K-98K context,
far below the 262144 threshold where DCA starts to matter, and they use a dummy
reward (timing/routing measurements only, no policy learning).

Idempotent and self-limiting: exits 0 with no write when the key is absent, so
run_test.sh can call it unconditionally on every launch. Writes a one-time
`config.json.dca-backup` next to the file the first time it strips anything.

Usage:
    strip_dca_config.py <model_dir_or_config_json> [more...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_KEY = "dual_chunk_attention_config"


def strip_one(target: Path) -> bool:
    """Strip the key from one model dir / config.json. True if the file changed."""
    config_path = target / "config.json" if target.is_dir() else target
    if not config_path.is_file():
        # Not an error: MODEL_PATH may be a bare HF repo id that is resolved from
        # the hub cache rather than a local directory. Nothing to patch.
        print(f"strip_dca_config: no config.json at {config_path} - skipping")
        return False

    with config_path.open() as fh:
        config = json.load(fh)

    if _KEY not in config:
        print(f"strip_dca_config: {config_path} already has no {_KEY} - no change")
        return False

    backup = config_path.with_suffix(".json.dca-backup")
    if not backup.exists():
        shutil.copy2(config_path, backup)
        print(f"strip_dca_config: saved original to {backup}")

    del config[_KEY]
    # indent=2 matches how HF writes these files; keeps the diff readable if
    # someone inspects the checkpoint by hand later.
    with config_path.open("w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    print(f"strip_dca_config: removed {_KEY} from {config_path}")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for raw in argv:
        strip_one(Path(raw).expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
