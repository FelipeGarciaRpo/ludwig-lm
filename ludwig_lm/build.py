"""The shard builder (T1): from the recipe's sources to the shards the training loop reads.

Stages, run by ``scripts/build_shards.py``:

  plan     every source and benchmark checked against the Hub: files, columns, the first document
  prepare  once: the decontamination set and Stack-Edu's permissive file lists
  val      the held-out documents of each source: the first ones of its last file, without tags
  train    the train shards, a few dozen per session; the state is saved after every shard, so a session
           that dies loses at most one shard, and the next one continues exactly
  samples  a few decoded documents per source, to read before the trunk starts

Each document goes through: its formatter (sources.py) → the key check against validation → data-mode
tokenization (chat sources with the chat template and ``<think>`` as id 4) → decontamination → FIM on 10%
of code → a shuffle buffer of its source. The mixer then takes the source furthest behind its share of
tokens, drops a page another source already used (by URL or Common Crawl record id), and the writer adds
the MeCo tag and EOS and cuts fixed-size shards.

Determinism: the same recipe, tokenizer, seed and builder version give the same shards, byte for byte,
whether built in one session or many. ``build_id`` names that combination.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from .data import load_tokenizer
from .shards import DTYPE, FORMAT, META_END, META_START, SHARD_TOKENS, TokenStream, sha256_file, write_tokens
from .sources import (FORMATTERS, ROWWISE, LocalHub, Permissive, columns_of, domain_of, get, iter_rows,
                      key64, normalize_url, select_files)

BUILDER_VERSION = "1"      # bump whenever the same recipe and tokenizer would give different shards
EOS, IM_START, IM_END, THINK, THINK_END = 0, 2, 3, 4, 5
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 10, 11, 12
NAMED = {"<|endoftext|>": EOS, "<|im_start|>": IM_START, "<|im_end|>": IM_END, "<think>": THINK,
         "</think>": THINK_END, "<|fim_prefix|>": FIM_PREFIX, "<|fim_middle|>": FIM_MIDDLE,
         "<|fim_suffix|>": FIM_SUFFIX, "<|reserved_0|>": META_START, "<|reserved_1|>": META_END}


def log(*a):
    print(*a, flush=True)


def sha1_json(obj) -> str:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return hashlib.sha1(json.dumps(clean(obj), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- recipe
def load_recipe(path) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [s["name"] for s in r["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("two sources share a name")
    total = sum(float(s["weight"]) for s in r["sources"])
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"the weights add up to {total}, not 100")
    for dom, srcs in r["eval_domains"].items():
        if missing := [s for s in srcs if s not in names]:
            raise ValueError(f"eval domain {dom}: unknown sources {missing}")
    for s in r["sources"]:
        if s["kind"] not in FORMATTERS:
            raise ValueError(f"{s['name']}: unknown kind {s['kind']!r}")
    r["_sha1"] = sha1_json(r)
    return r


class Coder:
    """The data-mode tokenizer plus the structure the pipeline adds around the text."""

    def __init__(self, tokenizer_path):
        self.path = Path(tokenizer_path)
        self.sha1 = hashlib.sha1(self.path.read_bytes()).hexdigest()
        self.tok = load_tokenizer(self.path)
        for s, i in NAMED.items():
            if self.tok.token_to_id(s) != i:
                raise ValueError(f"{self.path}: {s} is id {self.tok.token_to_id(s)}, the pipeline expects {i}")

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(e.ids, dtype=DTYPE) for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def chat(self, messages) -> np.ndarray | None:
        """ChatML, as ``apply_chat_template`` writes it, tokenized as the chat-mode tokenizer would: each text
        between control tokens is encoded whole, and the control tokens are ids. Only an assistant's
        ``<think>`` block may carry control strings; anything else that does is dropped."""
        parts = []
        for role, content in messages:
            if any(s in content for s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                return None
            if role != "assistant" and ("<think>" in content or "</think>" in content):
                return None
            if role == "assistant" and (content.count("<think>") > 1 or content.count("<think>") != content.count("</think>")):
                return None
            parts += [IM_START, f"{role}\n"]
            rest = content
            while "<think>" in rest:
                a, _, b = rest.partition("<think>")
                c, _, rest = b.partition("</think>")
                parts += [a, THINK, c, THINK_END]
            parts += [rest, IM_END, "\n"]
        texts = [p for p in parts if isinstance(p, str) and p]
        enc = iter(self.encode(texts))
        out = [np.array([p], dtype=DTYPE) if isinstance(p, int) else next(enc) for p in parts if not isinstance(p, str) or p]
        return np.concatenate(out)

    def fim(self, text: str, rng: random.Random) -> np.ndarray:
        """Fill in the middle, prefix-suffix-middle order (Qwen2.5-Coder's), split at two random characters."""
        i, j = sorted((rng.randint(0, len(text)), rng.randint(0, len(text))))
        pre, mid, suf = self.encode([text[:i], text[i:j], text[j:]])
        return np.concatenate([[FIM_PREFIX], pre, [FIM_SUFFIX], suf, [FIM_MIDDLE], mid]).astype(DTYPE)

    def tag(self, label: str, template: str = "URL: {}") -> np.ndarray:
        return np.concatenate([[META_START], self.encode([template.format(label)])[0], [META_END]]).astype(DTYPE)


# ----------------------------------------------------------------------------- decontamination
class Decon:
    """The benchmarks' 13-grams of tokens, as exact 64-bit hashes (sorted, one lookup per 13-gram).

    A document that shares one 13-gram with a benchmark item is dropped. 13-grams that appear in more than
    ``max_items`` different items are templates ("Which of the following…") and do not count."""

    N = 13
    _R = np.random.default_rng(20261115).integers(1, 2**63, size=1 << 16, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    _P = np.array([pow(0x9E3779B97F4A7C15, 12 - j, 1 << 64) for j in range(13)], dtype=np.uint64)

    def __init__(self, hashes: np.ndarray, bench: np.ndarray, names: list[str]):
        self.hashes, self.bench, self.names = hashes, bench, names

    @classmethod
    def ngrams(cls, ids: np.ndarray) -> np.ndarray:
        if len(ids) < cls.N:
            return np.zeros(0, dtype=np.uint64)
        return sliding_window_view(cls._R[np.asarray(ids, dtype=np.int64)], cls.N) @ cls._P

    def hit(self, ids: np.ndarray) -> int:
        """The index of a benchmark that shares a 13-gram with ``ids``, or -1."""
        if not len(self.hashes):
            return -1
        h = self.ngrams(ids)
        if not len(h):
            return -1
        i = np.minimum(np.searchsorted(self.hashes, h), len(self.hashes) - 1)
        found = np.nonzero(self.hashes[i] == h)[0]
        return int(self.bench[i[found[0]]]) if len(found) else -1

    @classmethod
    def build(cls, coder: Coder, hub, specs: list[dict], max_items: int = 10):
        all_h, all_item, names, report = [], [], [], {}
        item = 0
        for b, spec in enumerate(specs):
            files = select_files(hub.tree(spec["repo"], spec.get("revision")), spec)
            if not files:
                raise FileNotFoundError(f"benchmark {spec['name']}: no files in {spec['repo']} for {spec.get('prefix')!r}")
            texts, items = [], []
            for f in files:
                for _, row in iter_rows(hub, spec["repo"], f, spec.get("format", "parquet"), None, 0, spec.get("revision")):
                    for fld in spec["fields"]:
                        v = get(row, fld)
                        for t in (v if isinstance(v, list) else [v]):
                            if isinstance(t, str) and t.strip():
                                texts.append(t)
                                items.append(item)
                    item += 1
            n_ngrams = 0
            for k in range(0, len(texts), 512):
                for ids, it in zip(coder.encode(texts[k:k + 512]), items[k:k + 512]):
                    h = cls.ngrams(ids)
                    all_h.append(h)
                    all_item.append(np.full(len(h), it, dtype=np.int64))
                    n_ngrams += len(h)
            names.append(spec["name"])
            report[spec["name"]] = {"files": len(files), "items": 0, "texts": len(texts), "ngrams": n_ngrams,
                                    "_end_item": item}
        h = np.concatenate(all_h) if all_h else np.zeros(0, np.uint64)
        it = np.concatenate(all_item) if all_item else np.zeros(0, np.int64)
        # distinct items per 13-gram: templates shared by many items are not contamination
        pairs = np.unique(np.stack([h.view(np.int64), it]), axis=1) if len(h) else np.zeros((2, 0), np.int64)
        uh, first, counts = np.unique(pairs[0], return_index=True, return_counts=True)
        keep = counts <= max_items
        bounds = np.array([report[n]["_end_item"] for n in names])        # items of benchmark b end here
        bench_of = np.searchsorted(bounds, pairs[1][first[keep]], side="right").astype(np.uint8)
        hashes = uh[keep].view(np.uint64)
        order = np.argsort(hashes, kind="stable")
        prev = 0
        for n in names:
            end = report[n].pop("_end_item")
            report[n]["items"], prev = end - prev, end
        report["_total"] = {"ngrams": int(len(hashes)), "dropped_as_templates": int((~keep).sum()),
                            "max_items": max_items}
        return cls(hashes[order], bench_of[order], names), report

    def save(self, path):
        np.savez(path, hashes=self.hashes, bench=self.bench, names=np.array(self.names))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["hashes"], z["bench"], [str(n) for n in z["names"]])


# ----------------------------------------------------------------------------- documents in flight
class TokDoc:
    __slots__ = ("ids", "key", "url", "cc", "attr", "fim", "chat", "label")

    def __init__(self, ids, key, url, cc, attr, fim, chat, label):
        self.ids, self.key, self.url, self.cc, self.attr = ids, key, url, cc, attr
        self.fim, self.chat, self.label = fim, chat, label


class Owners:
    """Who used a page first, by URL and by Common Crawl record id, across the sources of a group: the same
    page never enters twice in two sources or two forms (original, rewrite, translation). A source may
    repeat its own pages in later epochs."""

    def __init__(self):
        self.url: dict[int, int] = {}
        self.cc: dict[int, int] = {}

    def taken(self, d: TokDoc, s: int) -> bool:
        return ((d.url is not None and self.url.get(d.url, s) != s) or
                (d.cc is not None and self.cc.get(d.cc, s) != s))

    def claim(self, d: TokDoc, s: int):
        if d.url is not None:
            self.url.setdefault(d.url, s)
        if d.cc is not None:
            self.cc.setdefault(d.cc, s)

    def save(self, path):
        np.savez(path, url_k=np.fromiter(self.url.keys(), np.uint64, len(self.url)),
                 url_v=np.fromiter(self.url.values(), np.uint8, len(self.url)),
                 cc_k=np.fromiter(self.cc.keys(), np.uint64, len(self.cc)),
                 cc_v=np.fromiter(self.cc.values(), np.uint8, len(self.cc)))

    @classmethod
    def load(cls, path):
        o, z = cls(), np.load(path)
        o.url = dict(zip(z["url_k"].tolist(), z["url_v"].tolist()))
        o.cc = dict(zip(z["cc_k"].tolist(), z["cc_v"].tolist()))
        return o


class SourceStream:
    """The documents of one source, tokenized, in the order of a seeded shuffle buffer.

    Files are read in a seeded order per epoch; when they run out, the next epoch reshuffles them (a source
    smaller than its share repeats; design 4.6). The cursor is (epoch, file, row, documents taken)."""

    def __init__(self, idx, spec, hub, coder, ctx, seed, decon, val_keys, buffer_docs, meta, state=None):
        self.idx, self.spec, self.hub, self.coder, self.decon = idx, spec, hub, coder, decon
        self.name, self.seed, self.size, self.meta = spec["name"], seed, buffer_docs, meta
        self.ctx = {**ctx, "seed": seed}
        self.files = select_files(hub.tree(spec["repo"], spec.get("revision")), spec)
        if not self.files:
            raise FileNotFoundError(f"{self.name}: no files in {spec['repo']} for {spec.get('prefix')!r}")
        self.rowwise = spec["kind"] in ROWWISE
        self.cols = columns_of(spec) if spec.get("format", "parquet") == "parquet" else None
        self.val_keys = val_keys if val_keys is not None else np.zeros(0, np.uint64)
        self.group = spec.get("group")
        st = state or {}
        self.cursor = dict(st.get("cursor") or {"epoch": 0, "file": 0, "row": 0, "docs": 0})
        self.rng = random.Random(f"{seed}/{self.name}/buffer")
        if st.get("rng"):
            self.rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
        self.buffer: list[TokDoc] = st.get("buffer", [])
        self.stats = Counter(st.get("stats", {}))
        self._gen = None

    def order(self, epoch: int) -> list[str]:
        files = list(self.files)
        random.Random(f"{self.seed}/{self.name}/files/{epoch}").shuffle(files)
        return files

    def _docs(self):
        empty_epochs = 0
        while True:
            files, got = self.order(self.cursor["epoch"]), 0
            while self.cursor["file"] < len(files):
                f = files[self.cursor["file"]]
                start = self.cursor["row"] if self.rowwise else 0
                skip = 0 if self.rowwise else self.cursor["docs"]
                rows = iter_rows(self.hub, self.spec["repo"], f, self.spec.get("format", "parquet"), self.cols, start,
                                 self.spec.get("revision"))
                for k, (nxt, doc) in enumerate(FORMATTERS[self.spec["kind"]](rows, self.spec, {**self.ctx, "file": f}), 1):
                    if k <= skip:
                        continue
                    self.cursor["row"], self.cursor["docs"] = (nxt if self.rowwise else 0), (
                        self.cursor["docs"] + 1 if self.rowwise else k)
                    got += 1
                    yield doc
                self.cursor.update(file=self.cursor["file"] + 1, row=0, docs=0)
            self.cursor.update(epoch=self.cursor["epoch"] + 1, file=0, row=0, docs=0)
            empty_epochs = empty_epochs + 1 if got == 0 else 0
            if empty_epochs >= 2:
                raise RuntimeError(f"{self.name}: a whole epoch gave no document (check its filters and fields)")

    def tokenize(self, docs, for_val=False) -> list[TokDoc]:
        """Tokenize, check against validation and the benchmarks, choose FIM, and drop what is too long."""
        sp, out = self.spec, []
        keyed = [(d, key64(f"{self.name}|{d.key}")) for d in docs]
        if not for_val and len(self.val_keys):
            keep = []
            for d, k in keyed:
                i = np.searchsorted(self.val_keys, np.uint64(k))
                if i < len(self.val_keys) and self.val_keys[i] == np.uint64(k):
                    self.stats["val_excluded"] += 1
                else:
                    keep.append((d, k))
            keyed = keep
        plain = [(d, k) for d, k in keyed if d.chat is None]
        ids_plain = self.coder.encode([d.text for d, _ in plain]) if plain else []
        made = [(d, k, ids) for (d, k), ids in zip(plain, ids_plain)]
        for d, k in keyed:
            if d.chat is not None:
                ids = self.coder.chat(d.chat)
                if ids is None:
                    self.stats["chat_malformed"] += 1
                    continue
                made.append((d, k, ids))
        for d, k, ids in made:
            if sp.get("max_tokens") and len(ids) > sp["max_tokens"]:
                self.stats["too_long"] += 1
                continue
            b = self.decon.hit(ids) if self.decon is not None else -1
            if b >= 0:
                self.stats["decon"] += 1
                self.stats[f"decon:{self.decon.names[b]}"] += 1
                continue
            fim = False
            if not for_val and d.code and sp.get("fim", 0) > 0:
                r = random.Random(f"{self.seed}/{self.name}/fim/{d.key}")
                if r.random() < sp["fim"] and len(d.text) >= 20:
                    ids, fim = self.coder.fim(d.text, r), True
            label = None
            if self.meta and not fim and d.chat is None and sp.get("tag"):
                label = (domain_of(d.url or d.attr.get("url")) or sp.get("tag_default")) if sp["tag"] == "url" else sp["tag"]
            url = key64(normalize_url(d.url)) if (self.group and d.url) else None
            cc = key64(f"cc|{d.cc_id}") if (self.group and d.cc_id) else None
            out.append(TokDoc(ids, d.key, url, cc, d.attr, fim, d.chat is not None, label))
            self.stats["fim" if fim else "docs"] += 1
        return out

    def fill(self):
        if self._gen is None:
            self._gen = self._docs()
        while len(self.buffer) < self.size:
            batch = list(islice(self._gen, 64))
            self.buffer.extend(self.tokenize(batch))

    def pop(self) -> TokDoc:
        self.fill()
        i = self.rng.randrange(len(self.buffer))
        self.buffer[i], self.buffer[-1] = self.buffer[-1], self.buffer[i]
        return self.buffer.pop()

    # ---- state
    def state(self) -> dict:
        st = self.rng.getstate()
        return {"cursor": self.cursor, "rng": [st[0], list(st[1]), st[2]], "stats": dict(self.stats)}


def _pack_buffers(streams) -> tuple[dict, list]:
    arrays, meta = {}, []
    for s in streams:
        b = s.buffer
        arrays[f"{s.idx}_ids"] = np.concatenate([d.ids for d in b]) if b else np.zeros(0, DTYPE)
        arrays[f"{s.idx}_len"] = np.array([len(d.ids) for d in b], dtype=np.int64)
        for f in ("url", "cc"):
            arrays[f"{s.idx}_{f}"] = np.array([getattr(d, f) or 0 for d in b], dtype=np.uint64)
            arrays[f"{s.idx}_has{f}"] = np.array([getattr(d, f) is not None for d in b], dtype=bool)
        arrays[f"{s.idx}_flags"] = np.array([[d.fim, d.chat] for d in b], dtype=bool).reshape(-1, 2)
        meta.append([{"key": d.key, "attr": d.attr, "label": d.label} for d in b])
    return arrays, meta


def _unpack_buffer(z, meta, i) -> list[TokDoc]:
    lens = z[f"{i}_len"]
    ids = np.split(z[f"{i}_ids"], np.cumsum(lens)[:-1]) if len(lens) else []
    out = []
    for j, a in enumerate(ids):
        out.append(TokDoc(a.astype(DTYPE), meta[i][j]["key"],
                          int(z[f"{i}_url"][j]) if z[f"{i}_hasurl"][j] else None,
                          int(z[f"{i}_cc"][j]) if z[f"{i}_hascc"][j] else None,
                          meta[i][j]["attr"], bool(z[f"{i}_flags"][j][0]), bool(z[f"{i}_flags"][j][1]),
                          meta[i][j]["label"]))
    return out


class Mixer:
    """Picks the source furthest behind its share of tokens (exact shares at every scale, no sampling
    noise), and drops a page that another source of its group already used."""

    def __init__(self, streams, weights, owners: Owners, emitted=None):
        self.streams, self.owners = streams, owners
        self.w = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
        self.emitted = np.asarray(emitted if emitted is not None else np.zeros(len(streams)), dtype=np.float64)

    def next(self) -> tuple[int, TokDoc]:
        s = int(np.argmax(self.w * (self.emitted.sum() + 1.0) - self.emitted))
        st = self.streams[s]
        while True:
            d = st.pop()
            if st.group and self.owners.taken(d, s):
                st.stats["dup"] += 1
                continue
            if st.group:
                self.owners.claim(d, s)
            return s, d


INDEX_FIELDS = ("url", "repo", "path", "license", "author")


class ShardWriter:
    def __init__(self, out_dir, shard_tokens, next_index, carry=None):
        self.out, self.S, self.k = Path(out_dir), shard_tokens, next_index
        self.buf = np.empty(shard_tokens, dtype=DTYPE)
        self.n, self.rows, self.by_source = 0, [], Counter()
        if carry is not None and len(carry):
            self.buf[:len(carry)] = carry
            self.n = len(carry)

    def add(self, ids: np.ndarray, row: dict) -> list[dict]:
        row["start"], row["tokens"] = self.n, len(ids)
        self.rows.append(row)
        self.by_source[row["source"]] += len(ids)
        done, pos = [], 0
        while pos < len(ids):
            take = min(len(ids) - pos, self.S - self.n)
            self.buf[self.n:self.n + take] = ids[pos:pos + take]
            self.n += take
            pos += take
            if self.n == self.S:
                done.append(self.flush())
        return done

    def flush(self) -> dict:
        name = f"train/shard_{self.k:05d}.bin"
        sha = write_tokens(self.out / name, self.buf[:self.n])
        cols = {"source": pa.array([r["source"] for r in self.rows], pa.string()).dictionary_encode(),
                "key": pa.array([str(r["key"]) for r in self.rows], pa.string()),
                "start": pa.array([r["start"] for r in self.rows], pa.int64()),
                "tokens": pa.array([r["tokens"] for r in self.rows], pa.int64()),
                "tag": pa.array([r["tag"] for r in self.rows], pa.int16()),
                "fim": pa.array([r["fim"] for r in self.rows], pa.bool_()),
                "chat": pa.array([r["chat"] for r in self.rows], pa.bool_())}
        for f in INDEX_FIELDS:
            cols[f] = pa.array([None if r["attr"].get(f) is None else str(r["attr"][f]) for r in self.rows], pa.string())
        pq.write_table(pa.table(cols), self.out / name.replace(".bin", ".idx.parquet"), compression="zstd")
        entry = {"name": name, "tokens": int(self.n), "sha256": sha, "docs": len(self.rows),
                 "by_source": dict(sorted(self.by_source.items()))}
        self.k, self.n, self.rows, self.by_source = self.k + 1, 0, [], Counter()
        return entry

    def carry(self) -> np.ndarray:
        return self.buf[:self.n].copy()


# ----------------------------------------------------------------------------- stages
def new_manifest(recipe, coder) -> dict:
    build_id = hashlib.sha1(f"{FORMAT}|{BUILDER_VERSION}|{recipe['_sha1']}|{coder.sha1}".encode()).hexdigest()[:16]
    return {"format": FORMAT, "build_id": build_id, "builder": BUILDER_VERSION, "recipe": recipe["name"],
            "recipe_sha1": recipe["_sha1"], "tokenizer_sha1": coder.sha1, "seed": recipe.get("seed", 0),
            "dtype": "uint16", "shard_tokens": int(recipe.get("shard_tokens", SHARD_TOKENS)),
            "meta": {"enabled": bool(recipe.get("meta", {}).get("enabled", True)), "start": META_START, "end": META_END,
                     "template": recipe.get("meta", {}).get("template", "URL: {}")},
            "eval_domains": recipe["eval_domains"],
            "sources": {s["name"]: {"weight": s["weight"], "domain": s["domain"], "license": s.get("license"),
                                    "repo": s["repo"]} for s in recipe["sources"]},
            "val": {}, "shards": [], "decon": {}}


def _prep_ctx(prep_dir, recipe) -> dict:
    perm = {}
    for s in recipe["sources"]:
        if s["kind"] == "code" and s["language"] not in perm:
            perm[s["language"]] = Permissive.load(Path(prep_dir) / f"permissive_{s['language']}.parquet")
    return {"permissive": perm}


def stage_plan(recipe, hub, coder, out, max_rows=4000) -> bool:
    """Checks every source and benchmark against the Hub without building anything."""
    rep, ok = {"sources": {}, "benchmarks": {}}, True
    for s in recipe["sources"]:
        r = rep["sources"][s["name"]] = {}
        try:
            tree = hub.tree(s["repo"], s.get("revision"))
            files = select_files(tree, s)
            r.update(files=len(files), bytes=sum(tree[f] for f in files), first=files[0] if files else None,
                     last=files[-1] if files else None)
            if not files:
                raise FileNotFoundError(f"no files under {s.get('prefix')!r}")
            first = next(iter_rows(hub, s["repo"], files[0], s.get("format", "parquet"), None, 0, s.get("revision")))[1]
            r["columns"] = {k: (sorted(v)[:20] if isinstance(v, dict) else type(v).__name__) for k, v in first.items()}
            ctx = {"seed": 0, "file": files[0]}
            if s["kind"] == "code":
                pt = hub.tree(s["permissive"]["repo"], s["permissive"].get("revision"))
                pf = select_files(pt, {"prefix": s["permissive"]["prefix"]})
                r["permissive_files"] = len(pf)
                if not pf:
                    raise FileNotFoundError(f"Stack-Edu: no files under {s['permissive']['prefix']!r}")
                blob = get(first, s.get("blob", "id|metadata.blob_id"))
                r["blob_id_example"] = blob
                ctx["permissive"] = {s["language"]: _AcceptAll()}
            rows = islice(iter_rows(hub, s["repo"], files[0], s.get("format", "parquet"), None, 0, s.get("revision")),
                          max_rows)
            doc = next((d for _, d in FORMATTERS[s["kind"]](rows, s, ctx)), None)
            if doc is None:
                raise ValueError(f"the formatter gave no document in the first {max_rows} rows")
            ids = coder.chat(doc.chat) if doc.chat is not None else coder.encode([doc.text])[0]
            text = doc.text if doc.chat is None else " | ".join(f"{a}: {b}" for a, b in doc.chat)
            r.update(example=text[:300], example_tokens=int(len(ids)), url=doc.url, cc_id=doc.cc_id, key=doc.key[:80],
                     attr={k: str(v)[:80] for k, v in doc.attr.items()}, status="ok")
        except Exception as e:  # noqa: BLE001 - the plan reports every problem at once
            ok = False
            r.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    for b in recipe.get("benchmarks", []):
        r = rep["benchmarks"][b["name"]] = {}
        try:
            files = select_files(hub.tree(b["repo"], b.get("revision")), b)
            if not files:
                raise FileNotFoundError(f"no files under {b.get('prefix')!r}")
            first = next(iter_rows(hub, b["repo"], files[0], b.get("format", "parquet"), None, 0, b.get("revision")))[1]
            missing = [f for f in b["fields"] if get(first, f) is None]
            r.update(files=len(files), first=files[0], columns=sorted(first)[:30], missing_fields=missing,
                     status="ok" if not missing else "error")
            ok = ok and not missing
        except Exception as e:  # noqa: BLE001
            ok = False
            r.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    rep["ok"] = ok
    write_json(Path(out) / "plan.json", rep)
    for kind in ("sources", "benchmarks"):
        for n, r in rep[kind].items():
            log(f"  {'✅' if r.get('status') == 'ok' else '❌'} {n}: " + (
                f"{r.get('files')} files, {r.get('bytes', 0) / 1e9:.1f} GB · {r.get('example_tokens', '')} tokens"
                if r.get("status") == "ok" else r.get("error", "") or f"missing {r.get('missing_fields')}"))
    return ok


class _AcceptAll:
    def get(self, blob):
        return {"repo": None, "path": None, "license": None} if blob else None


def stage_prepare(recipe, hub, coder, out):
    out = Path(out)
    (out / "prep").mkdir(parents=True, exist_ok=True)
    t = time.time()
    decon, rep = Decon.build(coder, hub, recipe.get("benchmarks", []), recipe.get("decon", {}).get("max_items", 10))
    decon.save(out / "prep" / "decon.npz")
    langs = {}
    for s in recipe["sources"]:
        if s["kind"] == "code" and s["language"] not in langs:
            log(f"  Stack-Edu {s['language']}: {s['permissive']['repo']}/{s['permissive']['prefix']}")
            p = Permissive.build(hub, s["permissive"], say=log)
            p.save(out / "prep" / f"permissive_{s['language']}.parquet")
            langs[s["language"]] = len(p)
            log(f"  Stack-Edu {s['language']}: {len(p):,} permissive files")
    write_json(out / "prep" / "prepare.json", {"decon": rep, "permissive_files": langs, "seconds": round(time.time() - t)})
    log(f"  decontamination: {rep['_total']['ngrams']:,} 13-grams from {len(decon.names)} benchmarks "
        f"({rep['_total']['dropped_as_templates']:,} dropped as templates)")


def stage_val(recipe, hub, coder, out, manifest):
    """The first documents of each source's last file, up to ``val_tokens``: never trained (the train stage
    drops them by key), without tags, without FIM."""
    out = Path(out)
    decon = Decon.load(out / "prep" / "decon.npz")
    ctx = _prep_ctx(out / "prep", recipe)
    keys = {}
    for i, s in enumerate(recipe["sources"]):
        st = SourceStream(i, s, hub, coder, ctx, recipe.get("seed", 0), decon, None, 0, meta=False)
        f = st.files[-1]
        rows = iter_rows(hub, s["repo"], f, s.get("format", "parquet"), st.cols, 0, s.get("revision"))
        docs = FORMATTERS[s["kind"]](rows, s, {**st.ctx, "file": f})
        need, got, chunks, ks = int(s.get("val_tokens", 2_000_000)), 0, [], []
        for batch in iter(lambda: list(islice(docs, 64)), []):
            for d in st.tokenize([x for _, x in batch], for_val=True):
                chunks += [d.ids, np.array([EOS], DTYPE)]
                ks.append(key64(f"{s['name']}|{d.key}"))
                got += len(d.ids) + 1
            if got >= need:
                break
        if not got:
            raise RuntimeError(f"{s['name']}: its last file gave no validation document")
        name = f"val/{s['name']}.bin"
        sha = write_tokens(out / name, np.concatenate(chunks))
        manifest["val"][s["name"]] = {"name": name, "tokens": int(got), "sha256": sha, "docs": len(ks), "file": f}
        keys[s["name"]] = np.unique(np.array(ks, dtype=np.uint64))
        log(f"  val {s['name']}: {got:,} tokens, {len(ks):,} documents from {f}")
    np.savez(out / "val" / "keys.npz", **keys)


def _load_state(d: Path, streams_specs):
    st = json.loads((d / "state" / "state.json").read_text(encoding="utf-8"))
    z = np.load(d / "state" / "buffers.npz")
    meta = json.loads((d / "state" / "buffers.json").read_text(encoding="utf-8"))
    for i, s in enumerate(streams_specs):
        st["streams"][s["name"]]["buffer"] = _unpack_buffer(z, meta, i)
    carry = np.load(d / "state" / "carry.npy")
    return st, carry, Owners.load(d / "state" / "owners.npz")


def _save_state(out: Path, manifest, streams, mixer, writer, stats):
    d = out / "state"
    d.mkdir(parents=True, exist_ok=True)
    arrays, meta = _pack_buffers(streams)
    np.savez(d / "buffers.npz.tmp.npz", **arrays)
    os.replace(d / "buffers.npz.tmp.npz", d / "buffers.npz")
    write_json(d / "buffers.json", meta)
    np.save(d / "carry.tmp.npy", writer.carry())
    os.replace(d / "carry.tmp.npy", d / "carry.npy")
    mixer.owners.save(d / "owners.tmp.npz")
    os.replace(d / "owners.tmp.npz", d / "owners.npz")
    write_json(out / "manifest.json", manifest)
    write_json(d / "state.json", {"build_id": manifest["build_id"], "shards": len(manifest["shards"]),
                                   "next_shard": writer.k, "emitted": mixer.emitted.tolist(),
                                   "streams": {s.name: s.state() for s in streams}, "stats": stats,
                                   "saved_at": time.time()})


def stage_train(recipe, hub, coder, out, prev=None, n_shards=56, max_hours=11.0):
    """Builds up to ``n_shards`` more train shards in ``out``, continuing from ``prev`` (the previous
    session's output) or from ``out`` itself."""
    t0, out = time.time(), Path(out)
    src = Path(prev) if prev else out
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    fresh = new_manifest(recipe, coder)
    if manifest["build_id"] != fresh["build_id"]:
        raise ValueError(f"{src}: build {manifest['build_id']}; this recipe, tokenizer and builder give "
                         f"{fresh['build_id']}. A build cannot change halfway")
    if src != out:                                       # each session's output carries what the next one needs
        for sub in ("prep", "val"):
            if (src / sub).exists() and not (out / sub).exists():
                shutil.copytree(src / sub, out / sub)
    decon = Decon.load(out / "prep" / "decon.npz")
    ctx = _prep_ctx(out / "prep", recipe)
    vk = np.load(out / "val" / "keys.npz")
    specs, seed = recipe["sources"], recipe.get("seed", 0)
    meta_on, tmpl = manifest["meta"]["enabled"], manifest["meta"]["template"]
    if (src / "state" / "state.json").exists():
        st, carry, owners = _load_state(src, specs)
    else:
        st, carry, owners = {"streams": {s["name"]: {} for s in specs}, "emitted": None, "next_shard": 0,
                             "stats": {}}, None, Owners()
    if st.get("next_shard", 0) != len(manifest["shards"]):
        raise ValueError(f"{src}: the state is at shard {st.get('next_shard')} and the manifest at {len(manifest['shards'])}")
    streams = [SourceStream(i, s, hub, coder, ctx, seed, decon, vk[s["name"]] if s["name"] in vk else None,
                            int(recipe.get("buffer_docs", 2048)), meta_on, st["streams"].get(s["name"]))
               for i, s in enumerate(specs)]
    mixer = Mixer(streams, [s["weight"] for s in specs], owners, st.get("emitted"))
    writer = ShardWriter(out, manifest["shard_tokens"], st.get("next_shard", 0), carry)
    tags: dict[str, np.ndarray] = {}
    stats = Counter(st.get("stats", {}))
    made, times, t_shard = 0, [], time.time()
    while made < n_shards:
        s, d = mixer.next()
        tag = np.zeros(0, DTYPE)
        if meta_on and d.label:
            if d.label not in tags:
                tags[d.label] = coder.tag(d.label, tmpl)
                if len(tags) > 200_000:
                    tags.clear()
            tag = tags[d.label]
        ids = np.concatenate([tag, d.ids, np.array([EOS], DTYPE)])
        mixer.emitted[s] += len(ids)                       # before any save: the state must include this document
        row = {"source": streams[s].name, "key": d.key, "tag": len(tag), "fim": d.fim, "chat": d.chat, "attr": d.attr}
        for entry in writer.add(ids, row):
            manifest["shards"].append(entry)
            made += 1
            times.append(time.time() - t_shard)
            t_shard = time.time()
            stats["tokens"] += entry["tokens"]
            _save_state(out, manifest, streams, mixer, writer, dict(stats))
            log(f"  shard {writer.k - 1:05d}: {entry['docs']:,} documents, {times[-1] / 60:.1f} min "
                f"({entry['tokens'] / times[-1] / 1e3:,.0f}K tokens/s)")
            if (time.time() - t0 + np.mean(times)) / 3600 > max_hours:
                log(f"  time limit: {made} shards this session")
                n_shards = made
            if made >= n_shards:
                break
    report = {"session_shards": made, "total_shards": len(manifest["shards"]), "hours": round((time.time() - t0) / 3600, 2),
              "minutes_per_shard": round(float(np.mean(times)) / 60, 2) if times else None,
              "tokens_by_source": {n.name: int(v) for n, v in zip(streams, mixer.emitted)},
              "share_by_source": {n.name: round(float(v / max(1, mixer.emitted.sum())) * 100, 3)
                                  for n, v in zip(streams, mixer.emitted)},
              "stats": {n.name: dict(n.stats) for n in streams}}
    write_json(out / f"report_{writer.k:05d}.json", report)
    return report


def stage_samples(manifest_path, coder, out, per_source=3, roots=(), seed=0):
    """A few documents of each source from the newest shards, decoded with their control tokens, to read
    before the trunk starts."""
    stream = TokenStream(manifest_path, roots, verify=False)
    rng = random.Random(seed)
    lines, have = [], Counter()
    for i in reversed(range(len(stream.shards))):
        idx_path = stream.files.find(stream.shards[i]["name"]).with_suffix("").with_suffix(".idx.parquet")
        t = pq.read_table(idx_path).to_pylist()
        rng.shuffle(t)
        for r in t:
            if have[r["source"]] >= per_source:
                continue
            pos = i * stream.shard_tokens + r["start"]
            ids = stream.read(pos, min(r["tokens"], 1200, stream.total - pos))
            text = coder.tok.decode(ids.tolist(), skip_special_tokens=False)
            lines.append(f"===== {r['source']} · shard {i} · key {r['key'][:60]} · {r['tokens']} tokens · "
                         f"tag {r['tag']} · fim {r['fim']} · chat {r['chat']} · {r.get('url') or ''}\n{text}\n")
            have[r["source"]] += 1
        if len(have) == len(stream.manifest["sources"]) and min(have.values()) >= per_source:
            break
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return dict(have)


def make_hub(local: str | None = None, token: str | None = None):
    from .sources import Hub
    return LocalHub(local) if local else Hub(token)


def verify(manifest_path, roots=()) -> list[str]:
    """Every shard's sha256 against the manifest; returns the names that fail."""
    s = TokenStream(manifest_path, roots, verify=False)
    bad = []
    for e in s.shards + list(s.manifest["val"].values()):
        try:
            if sha256_file(s.files.find(e["name"])) != e["sha256"]:
                bad.append(e["name"])
        except FileNotFoundError:
            bad.append(e["name"] + " (missing)")
    return bad
