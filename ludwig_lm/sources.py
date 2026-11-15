"""Where the documents of each source come from (T1): the Hub, its files, and how each kind of source
becomes a document.

Everything that depends on the real Hub (repository, files, column names) is data in the recipe
(``configs/data_v3.json``); ``scripts/build_shards.py plan`` checks it against the Hub before a build.
A formatter turns the rows of one file into documents:

* **row by row** (``plain``, ``finephrase``, ``stackexchange``, ``chat_solution``, ``code``): a document
  comes from one row, so a resume restarts at the row after the last document taken;
* **per file** (``stackoverflow``, ``oasst``, ``parallel``): a document joins several rows (a question and
  its accepted answer, a conversation, consecutive sentence pairs), so a resume replays the file from its
  start and skips the documents already taken. Deterministic either way.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:                                    # 3–10× faster on the JSON-lines sources (Common Pile, TransWeb-Edu)
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

RETRYABLE = (OSError, EOFError, TimeoutError)   # dropped connections; corrupt data is not retried


# ----------------------------------------------------------------------------- access
class Hub:
    """Files of Hugging Face datasets, read as streams: a parquet file by row groups and only the needed
    columns, a JSON-lines file (gzipped or not) line by line."""

    def __init__(self, token: str | None = None):
        self.token, self._api, self._fs, self._trees = token, None, None, {}

    def _fsys(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def tree(self, repo: str, revision: str | None = None) -> dict[str, int]:
        """Every file of the repository with its size in bytes."""
        if (repo, revision) not in self._trees:
            from huggingface_hub import HfApi
            api = self._api = self._api or HfApi(token=self.token)
            files = {}
            for x in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True):
                if hasattr(x, "size") and not hasattr(x, "tree_id"):
                    files[x.path] = int(x.size or 0)
            self._trees[(repo, revision)] = files
        return self._trees[(repo, revision)]

    def open(self, repo: str, path: str, revision: str | None = None):
        rev = f"@{quote(revision, safe='')}" if revision else ""
        return self._fsys().open(f"datasets/{repo}{rev}/{path}", "rb", block_size=4 << 20)


class LocalHub(Hub):
    """The same interface over a folder laid out as <root>/<owner>/<dataset>/<path> (tests, local copies)."""

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)

    def tree(self, repo, revision=None):
        base = self.root / repo
        return {str(p.relative_to(base)).replace("\\", "/"): p.stat().st_size
                for p in sorted(base.rglob("*")) if p.is_file()} if base.exists() else {}

    def open(self, repo, path, revision=None):
        return open(self.root / repo / path, "rb")


def select_files(tree: dict[str, int], spec: dict) -> list[str]:
    """The source's files, sorted: under ``prefix``, with the format's extension, optionally restricted to
    names containing one of ``contains`` and to a numeric range (``number_range``, the last number in the
    name, inclusive: the Common Pile shards of one language)."""
    fmt = spec.get("format", "parquet")
    ext = {"parquet": (".parquet",), "jsonl": (".jsonl", ".json"), "jsonl.gz": (".jsonl.gz", ".json.gz")}[fmt]
    out = [f for f in tree if f.startswith(spec.get("prefix", "")) and f.endswith(ext)]
    if spec.get("contains"):
        out = [f for f in out if any(c in f for c in spec["contains"])]
    if spec.get("number_range"):
        lo, hi = spec["number_range"]
        num = lambda f: int(re.findall(r"(\d+)", Path(f).name)[-1]) if re.findall(r"(\d+)", Path(f).name) else -1
        out = [f for f in out if lo <= num(f) <= hi]
    return sorted(out)


def _top_columns(paths) -> list[str]:
    return sorted({p.split(".")[0] for spec in paths if spec for alt in spec.split("|") for p in [alt]})


def _rows(hub, repo, path, fmt, columns, start, revision):
    fh = hub.open(repo, path, revision)
    try:
        if fmt == "parquet":
            pf = pq.ParquetFile(fh, buffer_size=4 << 20)
            md, starts = pf.metadata, [0]
            for g in range(md.num_row_groups):
                starts.append(starts[-1] + md.row_group(g).num_rows)
            g0 = max([g for g in range(md.num_row_groups) if starts[g] <= start], default=0)
            names = set(pf.schema_arrow.names)
            cols = [c for c in (columns or []) if c in names] or None
            pos = starts[g0] if md.num_row_groups else 0
            for batch in pf.iter_batches(batch_size=256, row_groups=list(range(g0, md.num_row_groups)), columns=cols):
                for row in batch.to_pylist():
                    if pos >= start:
                        yield pos, row
                    pos += 1
        else:
            stream = gzip.GzipFile(fileobj=fh) if path.endswith(".gz") else fh
            for pos, line in enumerate(stream):
                if pos < start:
                    continue
                try:
                    row = _loads(line)
                except ValueError:
                    continue                                   # a broken line is skipped the same way every time
                if isinstance(row, dict):
                    yield pos, row
    finally:
        fh.close()


def iter_rows(hub, repo, path, fmt="parquet", columns=None, start=0, revision=None, retries=5):
    """(row index, row) from row ``start``. If the connection drops, reopen and continue where it was."""
    nxt, fails = start, 0
    while True:
        try:
            for i, row in _rows(hub, repo, path, fmt, columns, nxt, revision):
                nxt = i + 1
                yield i, row
            return
        except RETRYABLE:
            fails += 1
            if fails > retries:
                raise
            time.sleep(min(60, 5 * fails))


def get(row, path, default=None):
    """A dotted field ("metadata.url"); "a|b" tries a, then b. JSON-encoded dicts are decoded on the way."""
    if not path:
        return default
    for alt in path.split("|"):
        cur, ok = row, True
        for part in alt.split("."):
            if isinstance(cur, (str, bytes)) and cur[:1] in ("{", b"{"):
                try:
                    cur = _loads(cur)
                except ValueError:
                    ok = False
                    break
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None and cur != "":
            return cur
    return default


def key64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def normalize_url(url: str) -> str:
    try:
        u = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (u.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    path = u.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{u.query}" if u.query else "")


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = (urlsplit(url.strip()).hostname or "").lower()
    except ValueError:
        return None
    return (host[4:] if host.startswith("www.") else host) or None


# ----------------------------------------------------------------------------- cleaning
AUTOGEN = ("auto-generated", "autogenerated", "automatically generated", "do not edit", "generated by")
LICENSE_WORDS = ("license", "licence", "copyright", "spdx-license-identifier", "permission is hereby granted",
                 "all rights reserved", "warranty")


def code_ok(text: str, row: dict, spec: dict) -> bool:
    """Code filters (the tokenizer's): size, minified or generated files."""
    if not 50 <= len(text) <= spec.get("max_chars", 100_000):
        return False
    if spec.get("lines", True):
        lmax, lavg, alnum = row.get("max_line_length"), row.get("avg_line_length"), row.get("alphanum_fraction")
        if lmax is None or lavg is None or alnum is None:
            lines = text.splitlines() or [""]
            lmax, lavg = max(map(len, lines)), sum(map(len, lines)) / len(lines)
            alnum = sum(map(str.isalnum, text)) / max(1, len(text))
        if lmax > 1000 or lavg > 100 or alnum < 0.25:
            return False
    head = "\n".join(text[:2000].splitlines()[:5]).lower()
    return not any(k in head for k in AUTOGEN)


_COMMENT = {"hash": (r"#", None), "c": (r"//", (r"/*", r"*/")), "sql": (r"--", (r"/*", r"*/")),
            "css": (None, (r"/*", r"*/")), "html": (None, (r"<!--", r"-->"))}


def strip_license_header(text: str, style: str | None) -> str:
    """Remove a license comment at the top of a file (after a shebang): line comments or one block
    comment, and in Python a module docstring, when they mention a license or a copyright."""
    if not style or style not in _COMMENT:
        return text
    line, block = _COMMENT[style]
    keep = ""
    body = text
    if body.startswith("#!"):
        first, _, body = body.partition("\n")
        keep = first + "\n"
    stripped = body.lstrip()
    lead = body[:len(body) - len(stripped)]
    removed = None
    if block and stripped.startswith(block[0]):
        end = stripped.find(block[1], len(block[0]))
        if end != -1:
            removed, rest = stripped[:end + len(block[1])], stripped[end + len(block[1]):]
    elif line and stripped.startswith(line):
        lines = stripped.split("\n")
        n = 0
        while n < len(lines) and (lines[n].lstrip().startswith(line) or not lines[n].strip()):
            n += 1
        removed, rest = "\n".join(lines[:n]), "\n" + "\n".join(lines[n:])
    if removed is not None and any(w in removed.lower() for w in LICENSE_WORDS):
        body = rest.lstrip("\n")
        lead = ""
    if style == "hash" and body.lstrip().startswith(('"""', "'''")):        # a license in the module docstring
        b = body.lstrip()
        q = b[:3]
        end = b.find(q, 3)
        if end != -1 and end < 5000 and any(w in b[:end].lower() for w in LICENSE_WORDS):
            body = b[end + 3:].lstrip("\n")
    return keep + lead + body


_SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "<KEY>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<KEY>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), "<KEY>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"), "<KEY>"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "<KEY>"),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), "<KEY>"),
    (re.compile(r"\b(?:sk|rk)_live_[0-9a-zA-Z]{20,}\b"), "<KEY>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"), "<KEY>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "<KEY>"),
]
_ASSIGN = re.compile(r"""(?i)(\b(password|passwd|pwd|secret|secret_key|client_secret|api_key|apikey|"""
                     r"""access_token|auth_token|private_key)\b["']?\s*[:=]\s*)(["'])([^"'\s]{6,})\3""")


def scrub_secrets(text: str) -> str:
    for rx, repl in _SECRETS:
        text = rx.sub(repl, text)
    return _ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(3)}"
                       f"{'<PASSWORD>' if m.group(2).lower() in ('password', 'passwd', 'pwd') else '<KEY>'}{m.group(3)}",
                       text)


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.([A-Za-z]{2,})\b")
_FILE_EXT = {"png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "css", "js", "ts", "json", "html", "htm",
             "woff", "woff2", "ttf", "eot", "otf", "mp4", "webm", "mp3", "pdf", "map"}   # "logo@2x.png" is a file
_IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])")


def _public_ip(ip: str) -> bool:
    a, b, *_ = (int(x) for x in ip.split("."))
    return not (a in (0, 10, 127, 255) or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31)
                or (a == 169 and b == 254) or a >= 224)


def scrub_pii(text: str) -> str:
    """Emails and public IPv4 addresses, with Common Pile's placeholders (The Stack v1 does not remove them)."""
    text = _EMAIL.sub(lambda m: m.group(0) if m.group(1).lower() in _FILE_EXT else "<EMAIL_ADDRESS>", text)
    return _IPV4.sub(lambda m: "<IP_ADDRESS>" if _public_ip(m.group(0)) else m.group(0), text)


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
             "table", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.pre = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag == "li":
            self.out.append("- ")
        if tag == "pre":
            self.pre += 1

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre = max(0, self.pre - 1)
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data if self.pre else re.sub(r"[ \t\r\n]+", " ", data))


def html_to_text(s: str) -> str:
    """Stack Exchange bodies come as HTML: plain text with code blocks kept verbatim."""
    if "<" not in s or not re.search(r"</?(p|pre|code|div|br|li|a|h\d)\b", s):
        return s
    p = _Text()
    p.feed(s)
    p.close()
    return re.sub(r"\n[ \t]*\n[\s]*\n+", "\n\n", "".join(p.out)).strip()


# ----------------------------------------------------------------------------- documents
@dataclass
class Doc:
    text: str | None = None            # a plain document
    chat: list | None = None           # [(role, content)], rendered with the chat template
    key: str = ""                      # stable within the source: excludes validation, names it in the index
    url: str | None = None
    cc_id: str | None = None           # Common Crawl's record id: crosses rewrites and translations of a page
    attr: dict = field(default_factory=dict)   # attribution: url, repo, path, license, author
    code: bool = False                 # eligible for FIM


class Permissive:
    """Stack-Edu's permissively licensed files of one language: blob_id → repository, path and licenses.
    Lookups use the first 64 bits of the blob_id (a SHA-1), sorted."""

    COLUMNS = ["blob_id", "license_type", "repo_name", "path", "detected_licenses"]

    def __init__(self, table: pa.Table):
        self.table = table
        self.keys = np.asarray(table.column("k").to_numpy(), dtype=np.uint64)

    @staticmethod
    def prefix(blob_id: str) -> int:
        return int(str(blob_id)[:16], 16)

    @classmethod
    def build(cls, hub: Hub, spec: dict) -> "Permissive":
        files = select_files(hub.tree(spec["repo"], spec.get("revision")), {"prefix": spec["prefix"]})
        if not files:
            raise FileNotFoundError(f"{spec['repo']}: no parquet files under {spec['prefix']!r}")
        parts = []
        for f in files:
            fh = hub.open(spec["repo"], f, spec.get("revision"))
            try:
                pf = pq.ParquetFile(fh, buffer_size=4 << 20)
                cols = [c for c in cls.COLUMNS if c in pf.schema_arrow.names]
                t = pf.read(columns=cols)
            finally:
                fh.close()
            t = t.filter(pc.equal(t.column("license_type"), "permissive"))
            if "detected_licenses" in t.column_names:
                lic = [", ".join(x) if isinstance(x, list) else (x or "") for x in t.column("detected_licenses").to_pylist()]
            else:
                lic = [""] * t.num_rows
            parts.append(pa.table({"k": pa.array([cls.prefix(b) for b in t.column("blob_id").to_pylist()], pa.uint64()),
                                   "blob_id": t.column("blob_id"),
                                   "repo": t.column("repo_name") if "repo_name" in t.column_names else pa.nulls(t.num_rows, pa.string()),
                                   "path": t.column("path") if "path" in t.column_names else pa.nulls(t.num_rows, pa.string()),
                                   "license": pa.array(lic, pa.string())}))
        table = pa.concat_tables(parts)
        return cls(table.take(pc.sort_indices(table, [("k", "ascending")])))

    def save(self, path):
        pq.write_table(self.table, path, compression="zstd")

    @classmethod
    def load(cls, path):
        return cls(pq.read_table(path))

    def __len__(self):
        return len(self.keys)

    def get(self, blob_id) -> dict | None:
        if not blob_id:
            return None
        k = np.uint64(self.prefix(blob_id))
        i = int(np.searchsorted(self.keys, k))
        if i >= len(self.keys) or self.keys[i] != k:
            return None
        row = self.table.slice(i, 1).to_pylist()[0]
        return {"repo": row["repo"], "path": row["path"], "license": row["license"], "blob_id": row["blob_id"]}


# ----------------------------------------------------------------------------- formatters
def _common(row, spec, text) -> Doc:
    url = get(row, spec.get("url"))
    cc = get(row, spec.get("cc_id"))
    key = get(row, spec.get("key")) or cc or url or hashlib.sha1(text[:4096].encode("utf-8", "replace")).hexdigest()
    attr = {k: get(row, v) for k, v in (spec.get("attr") or {}).items()}
    if url and "url" not in attr:
        attr["url"] = url
    return Doc(text=text, key=str(key), url=url, cc_id=str(cc) if cc else None,
               attr={k: v for k, v in attr.items() if v is not None})


def _clean_text(text, row, spec):
    if not isinstance(text, str):
        return None
    if spec.get("html"):
        text = html_to_text(text)
    if spec.get("code"):
        if not code_ok(text, row, spec):
            return None
        text = strip_license_header(text, spec.get("comments"))
        text = scrub_secrets(text)
    if spec.get("pii"):
        text = scrub_pii(text)
    if len(text.strip()) < spec.get("min_chars", 50) or len(text) > spec.get("max_chars", 1_000_000):
        return None
    return text


def f_plain(rows, spec, ctx):
    for i, row in rows:
        text = _clean_text(get(row, spec["text"]), row, spec)
        if text is not None:
            d = _common(row, spec, text)
            d.code = bool(spec.get("code"))
            yield i + 1, d


def f_finephrase(rows, spec, ctx):
    """FinePhrase: the first rewrite that finished; ``text`` is the original page and is not used."""
    for i, row in rows:
        for r in get(row, spec.get("text", "rollout_results")) or []:
            if isinstance(r, dict) and r.get("finish_reason") == "stop" and (r.get("text") or "").strip():
                text = _clean_text(r["text"], row, spec)
                if text is not None:
                    yield i + 1, _common(row, spec, text)
                break


def f_stackexchange(rows, spec, ctx):
    """A question and its best answers (up to 3 with a positive score), without Stack Overflow."""
    skip = tuple(spec.get("skip_sites", ["stackoverflow.com"]))
    for i, row in rows:
        meta = get(row, spec.get("site", "metadata"))
        url = meta[0] if isinstance(meta, list) and meta else (meta if isinstance(meta, str) else None)
        if url and any(s in url for s in skip):
            continue
        q = html_to_text(get(row, "question") or "")
        answers = sorted((a for a in (get(row, "answers") or []) if isinstance(a, dict) and (a.get("pm_score") or 0) >= 1),
                         key=lambda a: -(a.get("pm_score") or 0))[:3]
        if not q or not answers:
            continue
        text = q + "".join(f"\n\nAnswer:\n{html_to_text(a.get('text') or '')}" for a in answers)
        if len(text) < spec.get("min_chars", 200):
            continue
        d = _common(row, {**spec, "key": spec.get("key", "qid")}, text)
        d.url = url
        d.attr = {"url": url, "author": ", ".join(str(a.get("author_profile") or a.get("author") or "") for a in answers)}
        yield i + 1, d


def f_stackoverflow(rows, spec, ctx):
    """A question (score ≥ 1) and its accepted answer, joined inside each file (posts come in id order and
    an accepted answer is usually close to its question; pairs split across files are dropped)."""
    pending, cap, n = {}, spec.get("pending", 200_000), 0
    for i, row in rows:
        kind = int(get(row, "PostTypeId") or 0)
        if kind == 1 and get(row, "AcceptedAnswerId") and int(get(row, "Score") or 0) >= spec.get("min_score", 1):
            pending[int(get(row, "Id"))] = row
            if len(pending) > cap:
                pending.pop(next(iter(pending)))
        elif kind == 2:
            q = pending.get(int(get(row, "ParentId") or -1))
            if q is not None and int(get(q, "AcceptedAnswerId") or -2) == int(get(row, "Id")):
                del pending[int(get(q, "Id"))]
                title, qb, ab = get(q, "Title") or "", html_to_text(get(q, "Body") or ""), html_to_text(get(row, "Body") or "")
                text = f"{title}\n\n{qb}\n\nAnswer:\n{ab}" if title else f"{qb}\n\nAnswer:\n{ab}"
                if len(text) >= spec.get("min_chars", 200):
                    qid = int(get(q, "Id"))
                    yield i + 1, Doc(text=text, key=str(qid), url=f"https://stackoverflow.com/questions/{qid}",
                                     attr={"url": f"https://stackoverflow.com/questions/{qid}",
                                           "author": f"{get(q, 'OwnerUserId')}, {get(row, 'OwnerUserId')}",
                                           "license": f"{get(q, 'ContentLicense')}, {get(row, 'ContentLicense')}"})
                    n += 1


def f_chat_solution(rows, spec, ctx):
    """OpenMathReasoning: the problem as the user turn, the solution (with its <think> block) as the
    assistant turn. Whether it fits in 2K is checked after tokenizing (``max_tokens``)."""
    for i, row in rows:
        p, s = get(row, spec.get("problem", "problem")), get(row, spec.get("solution", "generated_solution"))
        if not (isinstance(p, str) and isinstance(s, str) and p.strip() and s.strip()):
            continue
        key = hashlib.sha1((p + "\x00" + s).encode("utf-8", "replace")).hexdigest()
        yield i + 1, Doc(chat=[("user", p.strip()), ("assistant", s.strip())], key=key,
                         attr={"author": get(row, "generation_model")})


def f_oasst(rows, spec, ctx):
    """OpenAssistant: per conversation tree, the path that always takes the best-ranked reply."""
    langs = set(spec.get("langs", ["en", "es"]))
    msgs, kids = {}, {}
    for _, row in rows:
        if row.get("deleted"):
            continue
        msgs[row["message_id"]] = row
        kids.setdefault(row.get("parent_id"), []).append(row["message_id"])
    roots = sorted((m for m in kids.get(None, []) if msgs[m].get("lang") in langs),
                   key=lambda m: (msgs[m].get("message_tree_id") or m))
    for n, root in enumerate(roots):
        path, cur = [msgs[root]], root
        while kids.get(cur):
            ranked = [c for c in kids[cur] if msgs[c].get("rank") is not None] or kids[cur]
            cur = min(ranked, key=lambda c: (msgs[c].get("rank") if msgs[c].get("rank") is not None else 1e9, c))
            path.append(msgs[cur])
        while path and path[-1].get("role") != "assistant":
            path.pop()
        if len(path) < 2:
            continue
        chat = [("user" if m.get("role") == "prompter" else "assistant", (m.get("text") or "").strip()) for m in path]
        if all(c for _, c in chat):
            yield n + 1, Doc(chat=chat, key=str(msgs[root].get("message_tree_id") or root))


def f_parallel(rows, spec, ctx):
    """English–Spanish sentence pairs, 8–32 consecutive ones per document, one sentence per line and its
    translation on the next; the direction (en→es or es→en) is drawn per document."""
    src, tgt = spec.get("langs", ["en", "es"])
    group, n = [], 0
    rng = random.Random(f"{ctx.get('seed', 0)}/{spec['name']}/{ctx.get('file', '')}")
    size = rng.randint(8, 32)
    for i, row in rows:
        tr = get(row, spec.get("pair", "translation")) or row
        a, b = tr.get(src) if isinstance(tr, dict) else None, tr.get(tgt) if isinstance(tr, dict) else None
        if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
            continue
        group.append((a.strip(), b.strip()))
        if len(group) == size:
            flip = rng.random() < 0.5
            text = "\n\n".join(f"{y}\n{x}" if flip else f"{x}\n{y}" for x, y in group)
            yield i + 1, Doc(text=text, key=hashlib.sha1(text[:512].encode()).hexdigest(), attr={"url": spec.get("site")})
            n += 1
            group, size = [], rng.randint(8, 32)


def f_code(rows, spec, ctx):
    """Stack-Edu's permissive files of one language, with their content from Common Pile, by blob_id."""
    table = ctx["permissive"][spec["language"]]
    for i, row in rows:
        blob = get(row, spec.get("blob", "id|metadata.blob_id"))
        attr = table.get(blob)
        if attr is None:
            continue
        text = _clean_text(get(row, spec.get("text", "text")), row, {**spec, "code": True})
        if text is None:
            continue
        url = get(row, "metadata.url") or (f"https://github.com/{attr['repo']}/blob/HEAD/{attr['path']}"
                                           if attr.get("repo") else None)
        lic = get(row, "metadata.license") or attr.get("license")
        yield i + 1, Doc(text=text, key=str(blob), url=None, code=True,
                         attr={"url": url, "repo": attr.get("repo"), "path": attr.get("path"), "license": lic})


FORMATTERS = {"plain": f_plain, "finephrase": f_finephrase, "stackexchange": f_stackexchange,
              "stackoverflow": f_stackoverflow, "chat_solution": f_chat_solution, "oasst": f_oasst,
              "parallel": f_parallel, "code": f_code}
ROWWISE = {"plain", "finephrase", "stackexchange", "chat_solution", "code"}


# columns a formatter reads besides the ones the recipe names (None: it needs every column)
FORMATTER_COLUMNS = {
    "stackexchange": ["question", "answers", "qid", "metadata"], "oasst": None, "finephrase": ["rollout_results"],
    "chat_solution": ["problem", "generated_solution", "generation_model"], "parallel": ["translation"],
    "stackoverflow": ["Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "Score", "Title", "Body", "OwnerUserId",
                      "ContentLicense"],
    "code": ["id", "text", "metadata"]}


def columns_of(spec: dict) -> list[str] | None:
    """The top-level columns a parquet source needs (None: all)."""
    if spec.get("columns"):
        return spec["columns"]
    extra = FORMATTER_COLUMNS.get(spec["kind"], [])
    if extra is None:
        return None
    paths = [spec.get(k) for k in ("text", "url", "cc_id", "key", "blob", "problem", "solution", "pair", "site")]
    paths += list((spec.get("attr") or {}).values()) + extra
    cols = _top_columns(paths)
    if spec.get("code"):
        cols += ["max_line_length", "avg_line_length", "alphanum_fraction"]
    return cols or None
