"""T1: builds the trunk's pre-tokenized shards (CPU only; on Kaggle, one CPU session per batch of shards).

    python scripts/build_shards.py plan    --out OUT                 # every source and benchmark against the Hub
    python scripts/build_shards.py build   --out OUT [--from PREV]   # prepare + val the first time, then shards
    python scripts/build_shards.py samples --manifest OUT/manifest.json --out OUT/muestras.txt
    python scripts/build_shards.py verify  --manifest OUT/manifest.json [--roots DIR ...]

The first ``build`` prepares the decontamination set and Stack-Edu's permissive lists, writes the
validation streams, and then builds train shards until ``--shards`` or ``--max-hours``. A later session
continues from ``--from``: the previous session's output (``auto`` finds the most advanced state under
``--roots``, by default /kaggle/input). Each session's output holds its new shards plus everything the
next session needs, so a Kaggle session output can become the next session's input.

``HF_TOKEN`` (environment) opens the gated datasets (The Stack v1, StarCoderData, Flores+).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"      # tokenizing is the builder's main cost: use every core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ludwig_lm.build import (Coder, load_recipe, log, make_hub, new_manifest, stage_plan,  # noqa: E402
                             stage_prepare, stage_samples, stage_train, stage_val, verify, write_json)


def code_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - the code may come without .git
        return None


def find_previous(roots) -> str | None:
    best = None
    for r in roots:
        for f in glob.glob(str(Path(r) / "**" / "state" / "state.json"), recursive=True):
            st = json.loads(Path(f).read_text(encoding="utf-8"))
            if best is None or st["shards"] > best[0]:
                best = (st["shards"], str(Path(f).parent.parent))
    return best and best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", default=str(ROOT / "configs" / "data_v3.json"))
        p.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "tokenizer.json"))
        p.add_argument("--out", required=True)
        p.add_argument("--local-hub", default="", help="a folder laid out like the Hub (tests)")
        if name == "build":
            p.add_argument("--from", dest="prev", default="", help="the previous session's output, or 'auto'")
            p.add_argument("--roots", nargs="*", default=["/kaggle/input"], help="where 'auto' looks")
            p.add_argument("--shards", type=int, default=56, help="train shards this session (56 × 256 MiB fit in "
                                                                   "Kaggle's 20 GB of output)")
            p.add_argument("--max-hours", type=float, default=11.0)
            p.add_argument("--allow-code-change", action="store_true",
                           help="continue a build with another commit of this repository (only for changes that "
                                "cannot alter the shards, such as documentation)")
    p = sub.add_parser("samples")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "tokenizer.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--per-source", type=int, default=3)
    p.add_argument("--roots", nargs="*", default=[])
    p = sub.add_parser("verify")
    p.add_argument("--manifest", required=True)
    p.add_argument("--roots", nargs="*", default=[])
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        bad = verify(a.manifest, a.roots)
        log("all files match the manifest" if not bad else f"{len(bad)} files do not match: {bad[:10]}")
        return 0 if not bad else 1
    if a.cmd == "samples":
        have = stage_samples(a.manifest, Coder(a.tokenizer), a.out, a.per_source, a.roots)
        log(f"{sum(have.values())} documents from {len(have)} sources -> {a.out}")
        return 0

    recipe, coder = load_recipe(a.recipe), Coder(a.tokenizer)
    hub = make_hub(a.local_hub or None, os.environ.get("HF_TOKEN"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "plan":
        ok = stage_plan(recipe, hub, coder, out)
        log("\nplan: every source and benchmark answers" if ok else "\nplan: fix the ❌ above in the recipe")
        return 0 if ok else 1

    prev = find_previous(a.roots) if a.prev == "auto" else (a.prev or None)
    if a.prev == "auto" and prev is None:
        log("no previous state under " + ", ".join(a.roots) + ": this is the first session")
    shutil.copy(a.recipe, out / "recipe.json")
    if prev is None and not (out / "manifest.json").exists():
        manifest = new_manifest(recipe, coder)
        manifest["builder_commit"] = code_commit()
        log(f"build {manifest['build_id']} ({recipe['name']}, tokenizer {coder.sha1[:12]})\n[prepare]")
        stage_prepare(recipe, hub, coder, out)
        manifest["decon"] = json.loads((out / "prep" / "prepare.json").read_text(encoding="utf-8"))["decon"]["_total"]
        log("[val]")
        stage_val(recipe, hub, coder, out, manifest)
        write_json(out / "manifest.json", manifest)
    elif prev:
        before = json.loads((Path(prev) / "manifest.json").read_text(encoding="utf-8")).get("builder_commit")
        now = code_commit()
        if before and now and before != now and not a.allow_code_change:
            raise SystemExit(f"the build started with commit {before[:12]}, this is {now[:12]}: the same code must "
                             f"build every shard (pin the commit in the notebook, or rebuild from scratch)")
        log(f"continuing from {prev}")
    log(f"[train] up to {a.shards} shards or {a.max_hours} h")
    rep = stage_train(recipe, hub, coder, out, prev, a.shards, a.max_hours)
    log(f"[end] {rep['session_shards']} shards this session, {rep['total_shards']} in total, {rep['hours']} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
