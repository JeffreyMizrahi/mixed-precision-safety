from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.features import count_blocks, extract_all_features
from shared.models import list_lm_linear_layers, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_features")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def default_output_path(model_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    safe = model_name.replace("/", "_")
    return repo_root / "src" / "results" / f"exp02_features_{safe}.csv"


def run_extraction(model, *, out_path: Path) -> list[dict]:
    layers = list_lm_linear_layers(model)
    n_blocks = count_blocks(layers)
    log.info("Extracting features for %d layers across %d blocks", len(layers), n_blocks)

    t0 = time.time()
    feats = extract_all_features(layers, n_blocks)
    log.info("Done in %.1fs", time.time() - t0)

    if not feats:
        log.error("No features extracted, something is wrong with the model structure")
        return []

    rows = [f.to_dict() for f in feats]
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s (%d rows)", out_path, len(rows))
    return rows


def main() -> int:
    args = parse_args()
    out_path = Path(args.output) if args.output else default_output_path(args.model)

    if args.resume and out_path.exists():
        log.info("Output %s already exists, skipping (--resume)", out_path)
        return 0

    log.info("Extracting features for %s -> %s", args.model, out_path)
    model, _ = load_model(args.model, device=args.device)
    run_extraction(model, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
