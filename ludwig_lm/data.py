"""Provisional data reader for T0.3/T2: parquet documents -> tokens + EOS -> windows of fixed length.

T1 replaces it with Ludwig 1's reader, extended to the weighted mix of the design (4.3). This one already
has the properties the trunk needs, so the T2 runs exercise them:

* **Deterministic:** the same files, seed and number of GPUs give the same windows on every run.
* **Exact resume:** the state is three integers (epoch, slice, token offset). A checkpoint restarts the
  stream at the next unseen token: nothing is skipped or repeated, and no row is re-read to get there.
* **No dropped tails:** documents are joined with EOS and cut into consecutive windows of ``seq`` tokens.
  A window may span two documents, and two slices.
* **Data mode:** special-token strings inside the raw text are encoded as text, never as control tokens.
* **Disjoint ranks:** rank r takes every world-th slice. Each row group is read once, and tokenization
  runs in a background thread.

A slice is up to ``rows_per_slice`` consecutive rows of one row group. Each epoch shuffles the row groups,
and the slices inside each one, with a seed made of the data seed and the epoch number.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import queue
import random
import threading
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # one thread per GPU: training needs ~1.7K tokens/s

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def expand(patterns) -> list[str]:
    """Files matching any of the globs, sorted, so the order never depends on the file system."""
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"no files match {list(patterns)}")
    return files


def load_tokenizer(path) -> Tokenizer:
    """The tokenizer in data mode: no text produces a control id. "<|endoftext|>" or "<think>" written in a
    web page stay text; the pipeline adds the control ids (EOS after each document; later FIM and chat).

    ``encode_special_tokens`` only covers added tokens marked special, and the 64K tokenizer keeps
    ``<think>``, ``<tool_call>`` and their closing tags as non-special (as Qwen3 does, for chat), so every
    added token is marked special first: the same as ``para_datos()`` in the tokenizer's own loader."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    for added in j.get("added_tokens", []):
        added["special"] = True
    tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    tok.encode_special_tokens = True
    return tok


def eos_id_of(tok: Tokenizer, default: int) -> int:
    i = tok.token_to_id("<|endoftext|>")
    return default if i is None else int(i)


def batch_hash(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()[:16]


class _Slices:
    """Row-group metadata of a set of parquet files, the order of each epoch, and the tokens of a slice."""

    def __init__(self, files, tok: Tokenizer, eos_id: int, rows_per_slice: int = 1024, column: str = "text"):
        self.files, self.tok, self.column = list(files), tok, column
        self.eos = np.array([int(eos_id)], dtype=np.int32)
        self.groups = []                                   # [(file index, row group, [(start, rows), ...])]
        for fi, f in enumerate(self.files):
            md = pq.ParquetFile(f).metadata
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                self.groups.append((fi, rg, [(s, min(rows_per_slice, n - s)) for s in range(0, n, rows_per_slice)]))
        self.n_slices = sum(len(s) for _, _, s in self.groups)
        self._files, self._cached = {}, (None, None)

    def order(self, seed: int, epoch: int, shuffle: bool = True) -> list[tuple]:
        """Every slice as (file, row group, start, rows), in the order of this epoch."""
        rng = random.Random(f"ludwig/{seed}/{epoch}")       # str seeds hash with SHA-512: stable across runs
        groups = list(range(len(self.groups)))
        if shuffle:
            rng.shuffle(groups)
        out = []
        for g in groups:
            fi, rg, slices = self.groups[g]
            idx = list(range(len(slices)))
            if shuffle:
                rng.shuffle(idx)
            out.extend((fi, rg, *slices[i]) for i in idx)
        return out

    def tokens(self, sl) -> np.ndarray:
        fi, rg, start, n = sl
        if self._cached[0] != (fi, rg):                     # one row group in memory at a time
            pf = self._files.get(fi)
            if pf is None:
                pf = self._files[fi] = pq.ParquetFile(self.files[fi])
            self._cached = ((fi, rg), pf.read_row_group(rg, columns=[self.column]).column(0))
        texts = [t for t in self._cached[1].slice(start, n).to_pylist() if t]
        parts = []
        for enc in self.tok.encode_batch(texts, add_special_tokens=False):
            if enc.ids:
                parts.append(np.asarray(enc.ids, dtype=np.int32))
                parts.append(self.eos)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


class PackedReader:
    """Micro-batches of ``micro`` windows of ``seq`` tokens for one rank, produced in a background thread.

    ``state_dict()`` describes the batches the training loop has taken (not the ones prefetched), so it can
    be saved at any step; ``PackedReader(..., state=...)`` continues from there.
    """

    def __init__(self, files, tok: Tokenizer, seq: int, micro: int, rank: int = 0, world: int = 1, seed: int = 0,
                 eos_id: int = 0, rows_per_slice: int = 1024, column: str = "text", state: dict | None = None,
                 prefetch: int = 16):
        self.files = list(files)
        self.src = _Slices(self.files, tok, eos_id, rows_per_slice, column)
        self.seq, self.micro, self.rank, self.world, self.seed = seq, micro, rank, world, seed
        self.ident = {"files": [Path(f).name for f in self.files], "bytes": [os.path.getsize(f) for f in self.files],
                      "seq": seq, "rank": rank, "world": world, "seed": seed, "rows_per_slice": rows_per_slice,
                      "column": column, "eos_id": int(eos_id)}
        if self.src.n_slices < world:
            raise ValueError(f"{self.src.n_slices} slices for {world} ranks: lower --rows-per-slice")
        if state is not None:
            diff = {k: (state.get(k), v) for k, v in self.ident.items() if state.get(k) != v}
            if diff:
                raise ValueError(f"the saved reader state is for other data or another layout: {diff}")
        self._start = {k: int((state or {}).get(k, 0)) for k in ("epoch", "k", "off")}
        self._taken = dict(self._start)
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, name=f"reader-rank{rank}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ producer side
    def _mine(self, epoch):
        return self.src.order(self.seed, epoch)[self.rank::self.world]

    def _windows(self, st):
        """(window, state after it), forever, starting at state ``st``."""
        epoch, k, off = st["epoch"], st["k"], st["off"]
        mine, cur, empty = self._mine(epoch), None, 0
        while True:
            parts, need = [], self.seq
            while need:
                if cur is None:
                    cur = self.src.tokens(mine[k])
                    empty = empty + 1 if len(cur) == 0 else 0
                    if empty > len(mine):
                        raise ValueError("every slice of this rank is empty")
                take = cur[off:off + need]
                if len(take):
                    parts.append(take)
                    need -= len(take)
                    off += len(take)
                if off >= len(cur):
                    k, off, cur = k + 1, 0, None
                    if k == len(mine):
                        epoch, k = epoch + 1, 0
                        mine = self._mine(epoch)
            yield np.concatenate(parts), {"epoch": epoch, "k": k, "off": off}

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def _produce(self):
        try:
            gen = self._windows(self._start)
            while not self._stop.is_set():
                windows, st = [], None
                for _ in range(self.micro):
                    w, st = next(gen)
                    windows.append(w)
                self._put((torch.from_numpy(np.stack(windows).astype(np.int64)), st))
        except BaseException as e:  # noqa: BLE001 - the training loop re-raises it
            self._put(e)

    # ------------------------------------------------------------------ consumer side
    def next(self) -> torch.Tensor:
        item = self._q.get()
        if isinstance(item, BaseException):
            raise RuntimeError(f"data reader (rank {self.rank}) failed") from item
        batch, self._taken = item
        return batch

    def state_dict(self) -> dict:
        return {**self._taken, **self.ident}

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=10)


def peek_batch(files, tok, seq, micro, rank, world, seed, eos_id, rows_per_slice, column, state) -> torch.Tensor:
    """The micro-batch a reader restarted from ``state`` gives first (for the resume check)."""
    r = PackedReader(files, tok, seq, micro, rank, world, seed, eos_id, rows_per_slice, column, state=state, prefetch=1)
    try:
        return r.next()
    finally:
        r.close()


def val_windows(files, tok: Tokenizer, seq: int, n_per_rank: int, rank: int, world: int, eos_id: int,
                rows_per_slice: int = 1024, column: str = "text"):
    """The first n_per_rank × world windows of the validation files, in file order; this rank's share.

    Returns (windows [n, seq] int64, UTF-8 bytes of their targets), for bits per byte.
    """
    src = _Slices(files, tok, eos_id, rows_per_slice, column)
    need, windows, buf = n_per_rank * world, [], np.zeros(0, dtype=np.int32)
    for sl in src.order(0, 0, shuffle=False):
        buf = np.concatenate([buf, src.tokens(sl)])
        while len(buf) >= seq and len(windows) < need:
            windows.append(buf[:seq])
            buf = buf[seq:]
        if len(windows) == need:
            break
    if len(windows) < need:
        raise ValueError(f"the validation files give {len(windows)} windows of {seq}; {need} are needed")
    mine = windows[rank::world]
    n_bytes = sum(len(tok.decode(w[1:].tolist(), skip_special_tokens=True).encode("utf-8")) for w in mine)
    return torch.from_numpy(np.stack(mine).astype(np.int64)), n_bytes
