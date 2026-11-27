"""Build a small local corpus for the CPU router lab: English prose (vim and git docs), Python code (transformers),
Perl docs/code, and the repo's own Spanish docs and data samples. Tokenized with the repo's 64K tokenizer, then
remapped to the V-2 most frequent ids (0 = EOS, 1 = everything else), so the output head is cheap on a CPU.

The paths are the ones of the Windows PC it ran on (Git for Windows' documentation, the PC's Python packages and
a T1 sample file next to the checkout); the corpus is not in the repo. On November 27, 2026 it gave 10.41M tokens,
3.51 characters per token, and the top 4,094 ids covered 91.8% of them.

    python prep_data.py <repo> <out_dir> [V]
"""
import random
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
V = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
random.seed(0)

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\n\s*\n\s*\n+")


def read(p, html=False):
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if html:
        t = TAG.sub("", t).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
        t = WS.sub("\n\n", t)
    return t


def gather(root, pattern, cap_mb, html=False):
    files = sorted(Path(root).rglob(pattern))
    random.shuffle(files)
    docs, size = [], 0
    for f in files:
        t = read(f, html)
        if len(t) < 500:
            continue
        for i in range(0, len(t), 6000):          # documents of ~6K characters
            docs.append(t[i:i + 6000])
        size += len(t)
        if size > cap_mb * 1e6:
            break
    return docs


sources = {
    "vim-docs": gather("C:/Program Files/Git/usr/share/vim/vim90/doc", "*.txt", 9),
    "git-docs": gather("C:/Program Files/Git/mingw64/share/doc/git-doc", "*.html", 9, html=True),
    "python": gather(repo / ".venv/Lib/site-packages/transformers/models", "*.py", 12),
    "perl": gather("C:/Program Files/Git/usr/share/perl5", "*.p[mo]*", 6),
    "repo-es": gather(repo / "docs", "*.md", 2) + gather(repo.parent / "t1-shards1", "muestras.txt", 1),
}
for k, v in sources.items():
    print(f"{k:10s} {len(v):6d} docs  {sum(map(len, v)) / 1e6:6.1f} MB", flush=True)

docs = [d for v in sources.values() for d in v]
random.shuffle(docs)
tok = Tokenizer.from_file(str(repo / "tokenizer/tokenizer.json"))
ids = []
for i in range(0, len(docs), 512):
    for e in tok.encode_batch(docs[i:i + 512], add_special_tokens=False):
        ids.append(np.asarray(e.ids, dtype=np.int64))
    print(f"tokenized {min(i + 512, len(docs))}/{len(docs)}", flush=True)
total = sum(len(x) for x in ids)
counts = np.zeros(65536, dtype=np.int64)
for x in ids:
    counts += np.bincount(x, minlength=65536)
order = np.argsort(-counts)
remap = np.ones(65536, dtype=np.int64)            # 1 = the rest
remap[order[:V - 2]] = np.arange(2, V)
covered = counts[order[:V - 2]].sum() / total
print(f"{total / 1e6:.2f}M tokens, {total and sum(map(len, docs)) / total:.2f} chars per token; "
      f"the top {V - 2} ids cover {covered:.1%}")
stream = np.concatenate([np.concatenate([remap[x], [0]]) for x in ids]).astype(np.int16)
n_val = 200_000
out.mkdir(parents=True, exist_ok=True)
np.save(out / f"val_{V}.npy", stream[:n_val])
np.save(out / f"train_{V}.npy", stream[n_val:])
print("saved", len(stream) - n_val, "train tokens and", n_val, "val tokens")
