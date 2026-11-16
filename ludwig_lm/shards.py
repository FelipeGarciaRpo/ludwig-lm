"""Pre-tokenized shards (T1): the files, the manifest, and the reader of the training loop.

A build is a directory (one per Kaggle dataset; a later session's directory holds the newer shards):

    manifest.json                   the whole build so far: train shards in order with their token counts
                                    and sha256, the validation streams, the recipe and the tokenizer
    train/shard_00000.bin           token ids as little-endian uint16, ``shard_tokens`` each (the last may be
                                    shorter). Documents are joined with EOS; one may continue into the next shard
    train/shard_00000.idx.parquet   one row per document that starts in the shard: source, key, offset,
                                    length, and its attribution (url, repository, path, license, author)
    val/<source>.bin                the held-out documents of each source, without metadata tags, with EOS

The training stream is the concatenation of the train shards. A global token offset ``t`` lives in shard
``t // shard_tokens``; windows of ``seq`` tokens are cut one after another, without dropping tails, as in
the provisional reader. Reading is a slice of a memmap: the GPU session never downloads, filters or
tokenizes, and an exact resume is a single integer.

**Metadata tags (MeCo).** A tagged document starts with ``META_START`` + "URL: domain" + ``META_END``.
MeCo computes the loss only on the document's tokens, so :func:`meta_mask` marks the span (markers
included) from the ids alone, also when a window starts or ends inside one. ``ShardReader(meta="strip")``
removes the spans instead, so the same shards can train without tags.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

FORMAT = "ludwig-shards-1"
SHARD_TOKENS = 1 << 27            # 134,217,728 tokens = 256 MiB: about one 12-hour session of the trunk
DTYPE = np.dtype("<u2")
META_START, META_END = 16, 17     # <|reserved_0|>, <|reserved_1|> of the 64K tokenizer
MAX_META = 256                    # a tag is far shorter: a span never starts more than this before its end


class OutOfData(RuntimeError):
    """The stream has fewer tokens than the next batch needs: the next shards are not attached."""


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_tokens(path, ids: np.ndarray) -> str:
    """Write ``ids`` as uint16 and return the file's sha256 (computed on the bytes written)."""
    arr = np.ascontiguousarray(ids, dtype=DTYPE)
    data = arr.tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def load_manifest(path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("format") != FORMAT:
        raise ValueError(f"{path}: format {m.get('format')!r}, this reader reads {FORMAT!r}")
    return m


class _Files:
    """Finds a build's files by name: next to the manifest, else under the search roots (Kaggle mounts each
    dataset in its own folder under /kaggle/input)."""

    def __init__(self, manifest_path, roots=()):
        self.base = Path(manifest_path).resolve().parent
        self.roots = [Path(r) for r in roots]
        self._index: dict[str, list[Path]] | None = None

    def find(self, name: str) -> Path:
        p = self.base / name
        if p.exists():
            return p
        if self._index is None:
            self._index = {}
            for r in self.roots:
                for f in glob.glob(str(r / "**" / "*.bin"), recursive=True):
                    self._index.setdefault(Path(f).name, []).append(Path(f))
        hits = [h for h in self._index.get(Path(name).name, []) if h.as_posix().endswith(name)]
        if not hits:
            raise FileNotFoundError(f"{name}: not next to the manifest nor under {', '.join(map(str, self.roots))}")
        return hits[0]


class TokenStream:
    """The train shards of a manifest as one array of tokens, read through memmaps (one per shard, opened
    when first needed and checked against the manifest's sha256 unless ``verify`` is off)."""

    def __init__(self, manifest_path, roots=(), verify: bool = True):
        self.path = Path(manifest_path)
        self.manifest = load_manifest(manifest_path)
        self.files = _Files(manifest_path, roots)
        self.shard_tokens = int(self.manifest["shard_tokens"])
        self.shards = self.manifest["shards"]
        for i, s in enumerate(self.shards[:-1]):
            if s["tokens"] != self.shard_tokens:
                raise ValueError(f"shard {i} has {s['tokens']} tokens; only the last may differ from {self.shard_tokens}")
        self.total = sum(int(s["tokens"]) for s in self.shards)
        self.verify = verify
        self._maps: dict[int, np.memmap] = {}

    def _map(self, i: int) -> np.memmap:
        if i not in self._maps:
            s = self.shards[i]
            path = self.files.find(s["name"])
            if os.path.getsize(path) != int(s["tokens"]) * DTYPE.itemsize:
                raise ValueError(f"{path}: {os.path.getsize(path)} bytes, the manifest says {s['tokens']} tokens")
            if self.verify and sha256_file(path) != s["sha256"]:
                raise ValueError(f"{path}: its sha256 is not the manifest's")
            self._maps[i] = np.memmap(path, dtype=DTYPE, mode="r")
        return self._maps[i]

    def read(self, start: int, n: int) -> np.ndarray:
        """Tokens [start, start + n) as int64, across shards."""
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if start < 0 or start + n > self.total:
            raise OutOfData(f"tokens {start:,}–{start + n:,} requested; the attached shards hold {self.total:,} "
                            f"(shard {(start + n) // self.shard_tokens} and later are missing)")
        out, pos = [], start
        while pos < start + n:
            i, off = divmod(pos, self.shard_tokens)
            take = min(start + n - pos, int(self.shards[i]["tokens"]) - off)
            out.append(np.asarray(self._map(i)[off:off + take], dtype=np.int64))
            pos += take
        return np.concatenate(out) if len(out) > 1 else out[0]


def meta_mask(ids, start: int = META_START, end: int = META_END):
    """True where a token is inside a metadata span, markers included. Works on [T] or [B, T] tensors or
    arrays, including spans that begin before or end after the window."""
    t = ids if isinstance(ids, torch.Tensor) else torch.from_numpy(np.asarray(ids, dtype=np.int64))
    squeeze = t.dim() == 1
    if squeeze:
        t = t.unsqueeze(0)
    s, e = (t == start).to(torch.int32), (t == end).to(torch.int32)
    T = t.shape[1]
    big = torch.full_like(s[:, :1], T)
    first_s = torch.where(s.any(1, keepdim=True), s.argmax(1, keepdim=True), big)
    first_e = torch.where(e.any(1, keepdim=True), e.argmax(1, keepdim=True), big)
    starts_inside = (first_e < first_s).to(torch.int32)                 # the window opens inside a span
    depth = s.cumsum(1) - (e.cumsum(1) - e) + starts_inside              # the end marker counts as inside
    inside = depth > 0
    return inside[0] if squeeze else inside


def loss_mask(ids: torch.Tensor) -> torch.Tensor:
    """For a batch [B, T]: True for the targets ids[:, 1:] that count in the loss (outside metadata)."""
    return ~meta_mask(ids)[:, 1:]


class ShardReader:
    """Micro-batches of ``micro`` windows of ``seq`` tokens for one rank.

    Each micro-step takes ``micro × world`` consecutive windows of the stream; rank r gets windows
    r, r + world, r + 2·world… The state is the global token offset of the next unread window, the same
    on every rank, so a checkpoint restarts at the next unseen token, and ``seq``, ``micro`` and the number
    of GPUs may change between sessions (the 2K → 4K step of the trunk).

    ``meta``: "mask" (MeCo: the tags stay and the training loop leaves them out of the loss), "keep" (they
    count like any token) or "strip" (they are removed and windows are cut from what remains).
    """

    def __init__(self, stream: TokenStream, seq: int, micro: int, rank: int = 0, world: int = 1,
                 state: dict | None = None, meta: str = "mask"):
        if meta not in ("mask", "keep", "strip"):
            raise ValueError(f"meta={meta!r}")
        self.stream, self.seq, self.micro, self.rank, self.world, self.meta = stream, seq, micro, rank, world, meta
        self.ident = {"build_id": stream.manifest["build_id"], "meta": meta}
        if state is not None:
            if state.get("build_id") != self.ident["build_id"]:
                raise ValueError(f"the saved reader state is for build {state.get('build_id')}, "
                                 f"these shards are build {self.ident['build_id']}")
            if state.get("meta", meta) != meta:
                raise ValueError(f"the run used meta={state.get('meta')!r}; this one asks for {meta!r}")
        self.offset = int((state or {}).get("token", 0))

    # ---- strip mode: windows from the stream without metadata spans
    def _inside_at(self, pos: int) -> bool:
        """Whether raw position ``pos`` starts inside a metadata span (the last marker before it opens one)."""
        if pos == 0:
            return False
        back = self.stream.read(max(0, pos - MAX_META), pos - max(0, pos - MAX_META))
        marks = np.nonzero((back == META_START) | (back == META_END))[0]
        return bool(len(marks)) and back[marks[-1]] == META_START

    def _take_stripped(self, pos: int, need: int) -> tuple[np.ndarray, int]:
        """``need`` tokens outside metadata from raw position ``pos``, and the raw position after them."""
        inside = self._inside_at(pos)
        kept, used = [], pos
        while sum(len(k) for k in kept) < need:
            want = need - sum(len(k) for k in kept)
            block = self.stream.read(used, min(want + 4096, self.stream.total - used))
            if not len(block):
                raise OutOfData("the stream ends before the next batch")
            head = np.array([META_START if inside else 0], dtype=block.dtype)
            m = meta_mask(np.concatenate([head, block])).numpy()[1:] if inside else meta_mask(block).numpy()
            keep = np.nonzero(~m)[0]
            if len(keep) >= want:
                kept.append(block[keep[:want]])
                used += int(keep[want - 1]) + 1
                break
            kept.append(block[keep])
            used += len(block)
            inside = bool(m[-1]) and block[-1] != META_END
        return np.concatenate(kept), used

    # ---- the batches
    def _batch_at(self, offset: int) -> tuple[torch.Tensor, int]:
        span = self.micro * self.world * self.seq
        if self.meta == "strip":
            flat, after = self._take_stripped(offset, span)
        else:
            flat, after = self.stream.read(offset, span), offset + span
        windows = flat.reshape(self.micro * self.world, self.seq)[self.rank::self.world]
        return torch.from_numpy(np.ascontiguousarray(windows)), after

    def next(self) -> torch.Tensor:
        batch, self.offset = self._batch_at(self.offset)
        return batch

    def peek(self, state: dict) -> torch.Tensor:
        """The micro-batch a reader restarted from ``state`` gives first (for the resume check)."""
        return self._batch_at(int(state["token"]))[0]

    def tokens_left(self) -> int:
        return self.stream.total - self.offset

    def state_dict(self) -> dict:
        return {"token": self.offset, **self.ident}

    def close(self):
        pass


def val_sets(manifest_path, tok, seq: int, n_per_rank: int, rank: int, world: int, roots=(),
             domains: dict | None = None, verify: bool = True) -> dict:
    """Validation windows per domain: {domain: (windows [n, seq] int64, UTF-8 bytes of their targets)}.

    A domain is one or more sources (the manifest's ``eval_domains``). Its windows come round-robin from
    the start of each source's held-out stream, so every evaluation reads the same tokens.
    """
    m = load_manifest(manifest_path)
    files = _Files(manifest_path, roots)
    domains = domains or m["eval_domains"]
    need = n_per_rank * world
    out = {}
    for dom, sources in domains.items():
        streams = []
        for src in sources:
            v = m["val"][src]
            path = files.find(v["name"])
            if verify and sha256_file(path) != v["sha256"]:
                raise ValueError(f"{path}: its sha256 is not the manifest's")
            streams.append(np.memmap(path, dtype=DTYPE, mode="r"))
        per = [len(s) // seq for s in streams]
        if sum(per) < need:
            raise ValueError(f"domain {dom}: its sources hold {sum(per)} windows of {seq}; {need} are needed")
        windows, k = [], [0] * len(streams)
        while len(windows) < need:
            for j, s in enumerate(streams):
                if k[j] < per[j] and len(windows) < need:
                    windows.append(np.asarray(s[k[j] * seq:(k[j] + 1) * seq], dtype=np.int64))
                    k[j] += 1
        mine = windows[rank::world]
        n_bytes = sum(len(tok.decode(w[1:].tolist(), skip_special_tokens=True).encode("utf-8")) for w in mine)
        out[dom] = (torch.from_numpy(np.stack(mine)), n_bytes)
    return out
