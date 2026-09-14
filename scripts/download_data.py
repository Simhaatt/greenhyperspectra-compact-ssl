#!/usr/bin/env python3
"""Download GreenHyperSpectra into ``data/raw/``.

The dataset is not redistributed with this repository. This helper fetches it
from the Hugging Face Hub into the layout the pipeline expects; see
``docs/DATA.md`` for the expected result and for the dataset's own licence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsi_paths import RAW_DIR  # noqa: E402

REPO_ID = "EvaCherif/GreenHyperSpectra"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID,
                        help="Hugging Face dataset repository id")
    parser.add_argument("--dest", type=Path, default=RAW_DIR,
                        help="destination directory (default: <root>/data/raw)")
    parser.add_argument("--labelled-only", action="store_true",
                        help="skip the 1.9 GB unlabelled split")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface-hub is required:  pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    ignore = ["*unlabeled*"] if args.labelled_only else None

    print(f"Downloading {args.repo_id} -> {args.dest}")
    if os.environ.get("HF_HOME"):
        print(f"  (cache: {os.environ['HF_HOME']})")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.dest),
        ignore_patterns=ignore,
    )

    print("\nDone. Verify with:\n    python src/inspect_dataset.py")
    print("If the layout differs from docs/DATA.md, move the files to match "
          "the expected names before running the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
