#!/usr/bin/env python3
"""CPU test of T1: the shard builder, the shard reader and the training loop on shards. No downloads.

    python scripts/test_shards_cpu.py            # ~2–4 min; prints PASS or what failed

1. Writes a small Hub in work/test_shards_cpu/hub/ with one source of each kind (web pages with URLs and
   Common Crawl ids, a rewrite and a translation of some of them, Wikipedia, Stack Exchange, Stack Overflow
   posts, Stack-Edu code joined by blob_id, HTML, parallel text, reasoning traces, OpenAssistant trees) and
   a benchmark. Planted on purpose: a benchmark question inside a web page, pages repeated across sources,
   an AWS key, a license header, a minified file, emails and IPs in HTML, a Stack Overflow question inside
   Stack Exchange, a template sentence shared by many benchmark items.
2. Builds the same 10 shards twice: in one session, and in two sessions (6 + 4) continuing from the
   first session's output. Every shard must be identical, byte for byte.
3. Checks what went into the shards: each source's share of tokens, no page twice across sources or forms,
   the benchmark page gone, validation never trained, tags, FIM, the chat template, the code and HTML clean-up.
4. Reader: exact resume, two ranks, windows across shard boundaries, the metadata mask, the strip mode.
5. Trains the tiny model on the shards with 2 processes: A runs 8 steps and saves before step 4; B resumes
   there and must equal A's steps 4–7 bit for bit; evaluations report bits per byte per domain.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "work" / "test_shards_cpu"
HUB = WORK / "hub"
TOKENIZER = ROOT / "tokenizer" / "tokenizer.json"
FAILS: list[str] = []

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

EN = ("the of and to in is was for on that with as by at from which this are be have it not model learning "
      "energy history science water language computer people world river city music garden planet").split()
ES = ("el la de que y en los se del las un por con no una su para es al lo como más pero sus año niño corazón "
      "canción también día país según así después mundo río ciudad jardín planeta").split()
PLANTED = ("A farmer had seventeen sheep and all but nine ran away into the forest so how many sheep does the "
           "farmer have left at the end of the day")
TEMPLATE = "Which of the following statements about the history of science is the most accurate one to choose"


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def para(r, words, a=20, b=70):
    return " ".join(r.choice(words) for _ in range(r.randint(a, b))).capitalize() + "."


def doc(r, words):
    return "\n\n".join(para(r, words) for _ in range(r.randint(1, 4)))


def parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def jsonl(path, rows, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)


def make_hub():
    shutil.rmtree(HUB, ignore_errors=True)
    r = random.Random(7)
    web_a = {}
    for f in range(3):
        rows = []
        for i in range(60):
            t = doc(r, EN)
            if f == 0 and i == 5:
                t += "\n\n" + PLANTED + "."                          # a benchmark question inside a page
            uid = f"<urn:uuid:a-{f}-{i}>"
            url = f"https://site{i % 7}.org/a/{f}/{i}"
            web_a[uid] = url
            rows.append({"id": uid, "text": t, "metadata": {"url": url, "score": 3.1}})
        parquet(HUB / "t/web-a" / f"data/{f:03d}.parquet", rows)
    ids = list(web_a)
    for f in range(3):
        rows = []
        for i in range(60):
            url = web_a[ids[(f * 60 + i) % len(ids)]] if i % 7 == 0 else f"https://www.other{i % 5}.com/b/{f}/{i}/"
            t = doc(r, EN) + (("\n\n" + TEMPLATE + ".") if i == 3 else "")
            rows.append({"text": t, "url": url, "metadata": {"WARC-Record-ID": f"<urn:uuid:b-{f}-{i}>"}})
        parquet(HUB / "t/web-b" / f"data/dclm-dedup/{f:03d}.parquet", rows)
    for f in range(2):
        rows = []
        for i in range(50):
            uid = ids[(f * 50 + i * 3) % len(ids)] if i % 3 == 0 else f"<urn:uuid:r-{f}-{i}>"
            rows.append({"id": uid, "rollout_results": [{"text": doc(r, EN) + " Q: why? A: because.",
                                                          "finish_reason": "stop" if i % 9 else "length"}]})
        parquet(HUB / "t/rewrite" / f"faq/{f:03d}.parquet", rows)
    for f in range(3):
        jsonl(HUB / "t/es-web" / f"es_chunk_{f:02d}.jsonl",
              [{"text_es": doc(r, ES) + " ¿Qué tal? ¡Olé!", "url": f"https://es{i % 4}.ejemplo.es/{f}/{i}"} for i in range(60)])
    for f in range(2):
        parquet(HUB / "t/wiki" / f"20231101.en/{f:03d}.parquet",
                [{"id": f"{f}-{i}", "url": f"https://en.wikipedia.org/wiki/P{f}_{i}", "title": f"P{i}",
                  "text": doc(r, EN)} for i in range(50)])
    rows = []
    for i in range(60):
        site = "stackoverflow.com" if i == 2 else "math.stackexchange.com"
        rows.append({"qid": i, "question": f"<p>{para(r, EN)}</p>", "metadata": [f"https://{site}/questions/{i}"],
                     "answers": [{"text": f"<p>{para(r, EN)}</p><pre><code>x = {i}\nprint(x)\n</code></pre>",
                                  "pm_score": 3, "author_profile": f"https://{site}/users/{i}"},
                                 {"text": "<p>meh</p>", "pm_score": 0, "author_profile": "u"}]})
    parquet(HUB / "t/se" / "data/math/000.parquet", rows[:30])
    parquet(HUB / "t/se" / "data/math/001.parquet", rows[30:])
    for f in range(2):
        posts, pid = [], f * 10_000
        for q in range(40):
            qid, aid = pid + 1, pid + 3
            posts += [{"Id": qid, "PostTypeId": 1, "AcceptedAnswerId": aid, "ParentId": None, "Score": 2,
                       "Title": f"How to {r.choice(EN)} in Python?", "Body": f"<p>{para(r, EN)}</p>",
                       "OwnerUserId": 10 + q, "ContentLicense": "CC BY-SA 4.0"},
                      {"Id": pid + 2, "PostTypeId": 2, "AcceptedAnswerId": None, "ParentId": qid, "Score": 0,
                       "Title": None, "Body": "<p>not this one</p>", "OwnerUserId": 99, "ContentLicense": "CC BY-SA 4.0"},
                      {"Id": aid, "PostTypeId": 2, "AcceptedAnswerId": None, "ParentId": qid, "Score": 5,
                       "Title": None, "Body": f"<p>{para(r, EN)}</p><pre><code>import os\n</code></pre>",
                       "OwnerUserId": 20 + q, "ContentLicense": "CC BY-SA 4.0"}]
            pid += 3
        parquet(HUB / "t/so" / f"stackoverflow-posts-{f:05d}.parquet", posts)
    perm, special = [], {}
    for f in range(1, 4):
        rows = []
        for i in range(60):
            blob = f"{r.getrandbits(160):040x}"
            body = "\n".join(f"def {r.choice(['suma', 'cargar', 'parse'])}_{j}(x, y={j}):\n    # {para(r, EN, 4, 9)}\n"
                             f"    return x * {j} + y\n" for j in range(r.randint(2, 6)))
            if f == 1 and i == 1:
                body = "# Copyright 2020 Someone\n# Licensed under the Apache License, Version 2.0\n\n" + body
                special["license"] = blob
            if f == 1 and i == 2:
                body += "\nAWS_KEY = 'AKIAABCDEFGHIJKLMNOP'\npassword = \"hunter2hunter2\"\n"
                special["secret"] = blob
            if f == 1 and i == 3:
                body = "var a=1;" * 400
                special["minified"] = blob
            rows.append({"id": blob, "text": body, "metadata": {"license": "MIT", "url": f"https://github.com/o/r{i}/blob/x.py"}})
            perm.append({"blob_id": blob, "license_type": "permissive" if i % 10 else "no_license",
                         "repo_name": f"o/r{i}", "path": f"src/m{i}.py", "detected_licenses": ["MIT"]})
        jsonl(HUB / "t/cp" / f"cp-{f:04d}.json.gz", rows, gz=True)
    parquet(HUB / "t/stack-edu" / "Python/train-000.parquet", perm)
    for f in range(2):
        parquet(HUB / "t/stack-v1" / f"data/html/{f:03d}.parquet",
                [{"hexsha": f"{r.getrandbits(160):040x}", "max_stars_repo_name": f"o/web{i}", "max_stars_repo_path": "i.html",
                  "max_stars_repo_licenses": ["MIT"],
                  "content": f"<html><body><p>{para(r, EN)}</p>\n<p>write to jane.doe{i}@example.com</p>\n"
                             f"<img src=\"img/logo@2x.png\">\n<!-- server 8.8.{i}.4 and 192.168.0.{i} -->\n</body></html>\n"}
                 for i in range(40)])
    for f in range(2):
        parquet(HUB / "t/par" / f"en-es/{f:03d}.parquet",
                [{"translation": {"en": para(r, EN, 6, 14), "es": para(r, ES, 6, 14)}} for _ in range(240)])
    for f in range(2):
        rows = []
        for i in range(50):
            sol = f"<think>\n{para(r, EN, 10, 40)}\n</think>\n\nThe answer is {i}."
            if i % 10 == 0:
                sol = f"<think>\n{para(r, EN, 4000, 4200)}\n</think>\n\nLong."          # does not fit in max_tokens
            rows.append({"problem": f"Compute {i} plus {f}. " + para(r, EN, 5, 15), "generated_solution": sol,
                         "generation_model": "DeepSeek-R1"})
        parquet(HUB / "t/omr" / f"data/cot-{f:03d}.parquet", rows)
    for f, split in enumerate(("train", "validation")):
        msgs = []
        for t in range(12):
            lang = ["en", "es", "de"][t % 3]
            root = f"{split}-{t}"
            msgs.append({"message_id": root, "parent_id": None, "message_tree_id": root, "text": para(r, EN if lang != "es" else ES),
                         "role": "prompter", "lang": lang, "rank": None, "deleted": False})
            for k in range(2):
                msgs.append({"message_id": f"{root}-a{k}", "parent_id": root, "message_tree_id": root,
                             "text": para(r, EN if lang != "es" else ES), "role": "assistant", "lang": lang, "rank": k, "deleted": False})
        parquet(HUB / "t/oa" / f"data/{split}-000.parquet", msgs)
    items = [{"question": PLANTED, "answer": "Nine."}] + [{"question": f"{TEMPLATE}? Item {k}.", "answer": "B"} for k in range(12)]
    items += [{"question": para(r, EN, 15, 25), "answer": str(k)} for k in range(30)]
    parquet(HUB / "t/bench" / "main/test-000.parquet", items)
    return special


def recipe() -> dict:
    S = lambda **k: k  # noqa: E731
    return {
        "name": "test", "seed": 0, "shard_tokens": 8192, "buffer_docs": 16, "meta": {"enabled": True},
        "decon": {"max_items": 10},
        "eval_domains": {"en": ["web-a", "web-b"], "es": ["es-web"], "python": ["py"]},
        "sources": [
            S(name="web-a", domain="en", weight=20, repo="t/web-a", prefix="data/", kind="plain", text="text",
              url="metadata.url", cc_id="id", key="id", group="cc-en", tag="url", min_chars=50, val_tokens=1500),
            S(name="web-b", domain="en", weight=14, repo="t/web-b", prefix="data/dclm-dedup/", kind="plain", text="text",
              url="url", cc_id="metadata.WARC-Record-ID", group="cc-en", tag="url", min_chars=50, val_tokens=1500),
            S(name="rewrite", domain="en", weight=6, repo="t/rewrite", prefix="faq/", kind="finephrase",
              text="rollout_results", cc_id="id", group="cc-en", tag="finephrase", min_chars=50, val_tokens=600),
            S(name="es-web", domain="es", weight=14, repo="t/es-web", prefix="es_chunk", format="jsonl", kind="plain",
              text="text_es", url="url", tag="url", min_chars=50, val_tokens=1500),
            S(name="wiki", domain="en", weight=6, repo="t/wiki", prefix="20231101.en/", kind="plain", text="text",
              url="url", key="id", tag="url", min_chars=50, val_tokens=600),
            S(name="se", domain="en", weight=4, repo="t/se", prefix="data/", kind="stackexchange", key="qid",
              site="metadata", tag="url", min_chars=50, val_tokens=600),
            S(name="so", domain="code", weight=4, repo="t/so", prefix="stackoverflow-posts-", kind="stackoverflow",
              tag="stackoverflow.com", min_chars=50, val_tokens=600),
            S(name="py", domain="code", weight=12, repo="t/cp", prefix="", format="jsonl.gz", number_range=[1, 3],
              kind="code", language="python", blob="id", text="text", comments="hash", fim=0.3, tag="github.com",
              permissive={"repo": "t/stack-edu", "prefix": "Python/"}, val_tokens=1500),
            S(name="html", domain="code", weight=4, repo="t/stack-v1", prefix="data/html/", kind="plain", text="content",
              key="hexsha", code=True, pii=True, comments="html", tag="github.com",
              attr={"repo": "max_stars_repo_name", "path": "max_stars_repo_path"}, val_tokens=600),
            S(name="par", domain="es", weight=4, repo="t/par", prefix="en-es/", kind="parallel", pair="translation",
              langs=["en", "es"], site="un.org", tag="un.org", val_tokens=600),
            S(name="omr", domain="instruct", weight=8, repo="t/omr", prefix="data/cot-", kind="chat_solution",
              max_tokens=2048, val_tokens=600),
            S(name="oa", domain="instruct", weight=4, repo="t/oa", prefix="data/", kind="oasst", langs=["en", "es"],
              max_tokens=2048, val_tokens=300),
        ],
        "benchmarks": [{"name": "bench", "repo": "t/bench", "prefix": "main/test", "fields": ["question", "answer"]}],
    }


def run_build(args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "build_shards.py"), *args], cwd=ROOT,
                          capture_output=True, text=True)


def index_rows(out_dirs):
    rows = []
    for d in out_dirs:
        for f in sorted(Path(d).glob("train/shard_*.idx.parquet")):
            shard = int(f.name[6:11])
            rows += [dict(r, shard=shard) for r in pq.read_table(f).to_pylist()]
    return rows


def test_builder(special):
    from ludwig_lm.build import Coder, Decon, key64
    from ludwig_lm.shards import META_END, META_START, TokenStream, load_manifest

    rec = WORK / "recipe.json"
    rec.write_text(json.dumps(recipe(), indent=1), encoding="utf-8")
    common = ["--recipe", str(rec), "--tokenizer", str(TOKENIZER), "--local-hub", str(HUB)]
    print("builder: plan, one session of 10 shards, and 6 + 4 in two sessions", flush=True)
    p = run_build(["plan", *common, "--out", str(WORK / "plan")])
    plan = json.loads((WORK / "plan" / "plan.json").read_text())
    check(p.returncode == 0 and plan["ok"], "plan: every source and the benchmark answer")
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
    outs = {k: WORK / k for k in ("one", "s1", "s2")}
    for o in outs.values():
        shutil.rmtree(o, ignore_errors=True)
    runs = [run_build(["build", *common, "--out", str(outs["one"]), "--shards", "10"]),
            run_build(["build", *common, "--out", str(outs["s1"]), "--shards", "6"]),
            run_build(["build", *common, "--out", str(outs["s2"]), "--from", "auto", "--roots", str(outs["s1"]),
                       "--shards", "4"])]
    for r in runs:
        if r.returncode:
            print(r.stdout[-4000:], r.stderr[-4000:])
    check(all(r.returncode == 0 for r in runs), "the three builds finish")
    if any(r.returncode for r in runs):
        return None
    m1, m2 = load_manifest(outs["one"] / "manifest.json"), load_manifest(outs["s2"] / "manifest.json")
    check(len(m1["shards"]) == 10 and len(m2["shards"]) == 10, "10 shards each way")
    check([s["sha256"] for s in m1["shards"]] == [s["sha256"] for s in m2["shards"]],
          "in one session or in two, every shard is identical byte for byte")
    check(m1["val"] == m2["val"] and m1["build_id"] == m2["build_id"], "same validation streams and build id")
    from ludwig_lm.build import verify
    check(not verify(outs["s2"] / "manifest.json", [outs["s1"]]), "every file matches its sha256 (shards in two folders)")

    coder = Coder(TOKENIZER)
    stream = TokenStream(outs["one"] / "manifest.json")
    rows = index_rows([outs["one"]])
    toks = Counter()
    for r in rows:
        toks[r["source"]] += r["tokens"]
    total = sum(toks.values())
    w = {s["name"]: s["weight"] for s in recipe()["sources"]}
    off = {k: abs(100 * toks[k] / total - w[k]) for k in w}
    check(max(off.values()) < 1.5, f"each source's share is its weight within 1.5 points (worst {max(off, key=off.get)} "
                                   f"{max(off.values()):.2f})")
    group = [r for r in rows if r["source"] in ("web-a", "web-b", "rewrite")]
    owner = {}
    clash = 0
    for r in group:
        if r.get("url"):
            clash += owner.setdefault(r["url"], r["source"]) != r["source"]
    check(clash == 0, "no URL in two sources of the group")
    ids_a = {r["key"] for r in rows if r["source"] == "web-a"}
    ids_rw = {r["key"] for r in rows if r["source"] == "rewrite"}
    check(not (ids_a & ids_rw), "a page and its rewrite never both go in (Common Crawl id)")
    stats = json.loads(sorted((outs["one"]).glob("report_*.json"))[-1].read_text())["stats"]
    check(stats["web-b"].get("dup", 0) > 0 and stats["rewrite"].get("dup", 0) > 0, "repeated pages were dropped")
    check(not any(r["key"] == "<urn:uuid:a-0-5>" for r in rows) and stats["web-a"].get("decon:bench", 0) >= 1,
          "the page with the benchmark question is gone")
    tmpl_kept = any(TEMPLATE in coder.tok.decode(stream.read(r["shard"] * 8192 + r["start"], min(r["tokens"], stream.total - r["shard"] * 8192 - r["start"])).tolist())
                    for r in rows if r["source"] == "web-b")
    check(tmpl_kept, "a template sentence shared by many benchmark items does not count as contamination")
    vk = np.load(outs["one"] / "val" / "keys.npz")
    leak = [r for r in rows if r["source"] in vk and np.isin(np.uint64(key64(f"{r['source']}|{r['key']}")), vk[r["source"]])]
    check(not leak, "no validation document is in the train shards")

    def tokens_of(r, n=None):
        start = r["shard"] * 8192 + r["start"]
        return stream.read(start, min(n or r["tokens"], stream.total - start))

    tagged = [r for r in rows if r["tag"] > 0]
    check(tagged and all(tokens_of(r, r["tag"])[0] == META_START and tokens_of(r, r["tag"])[-1] == META_END for r in tagged[:50]),
          "tagged documents open with <|reserved_0|> URL: … <|reserved_1|>")
    t0 = next(r for r in tagged if r["source"] == "web-a")
    check(coder.tok.decode(tokens_of(t0, t0["tag"])[1:-1].tolist()).startswith("URL: site"), "the tag is the page's domain")
    check(all(r["tag"] == 0 for r in rows if r["chat"] or r["fim"]), "chat and FIM documents carry no tag")
    ends = [tokens_of(r)[-1] for r in rows if r["shard"] * 8192 + r["start"] + r["tokens"] <= stream.total]
    check(all(e == 0 for e in ends), "every document ends with EOS")
    fims = [r for r in rows if r["fim"]]
    check(fims and all(list(tokens_of(r, 1)) == [10] for r in fims) and all(12 in tokens_of(r) and 11 in tokens_of(r) for r in fims),
          "FIM documents: <|fim_prefix|> … <|fim_suffix|> … <|fim_middle|> …")
    r0 = random.Random(3)
    text = "def f(x):\n    return x + 1  # ñandú\n"
    f_ids = coder.fim(text, r0)
    i_suf, i_mid = int(np.where(f_ids == 12)[0][0]), int(np.where(f_ids == 11)[0][0])
    pre, suf, mid = (coder.tok.decode(f_ids[a:b].tolist()) for a, b in ((1, i_suf), (i_suf + 1, i_mid), (i_mid + 1, None)))
    check(pre + mid + suf == text, "FIM pieces put back together give the original file")
    chats = [r for r in rows if r["source"] == "omr"]
    c = tokens_of(chats[0])
    check(c[0] == 2 and 4 in c and 5 in c and list(c[-3:]) [-1] == 0 and 3 in c,
          "reasoning traces: <|im_start|>user … <|im_start|>assistant <think> … </think> … <|im_end|>, then EOS")
    check(all(r["tokens"] <= 2050 for r in chats) and stats["omr"].get("too_long", 0) > 0, "traces longer than 2K are dropped")
    trees = {int(r["key"].rsplit("-", 1)[1]) % 3 for r in rows if r["source"] == "oa"}   # tree t: en, es, de by t % 3
    check(trees == {0, 1}, "OpenAssistant: the English and Spanish conversations go in, the German ones do not")
    py = [coder.tok.decode(tokens_of(r).tolist()) for r in rows if r["source"] == "py"]
    check(not any("AKIA" in t or "hunter2" in t for t in py), "keys and passwords are gone from the code")
    check(any("<KEY>" in t for t in py) or special["secret"] not in {r["key"] for r in rows if r["source"] == "py"},
          "the key's file went in with <KEY> (or was not drawn)")
    check(not any("Licensed under the Apache" in t for t in py), "license headers are stripped")
    check(special["minified"] not in {r["key"] for r in rows}, "the minified file is dropped")
    keys_py = {r["key"] for r in rows if r["source"] == "py"}
    perm = pq.read_table(HUB / "t/stack-edu/Python/train-000.parquet").to_pylist()
    nonperm = {p["blob_id"] for p in perm if p["license_type"] != "permissive"}
    check(keys_py and not (keys_py & nonperm), "only Stack-Edu's permissive files enter")
    html = [coder.tok.decode(tokens_of(r).tolist()) for r in rows if r["source"] == "html"]
    check(html and not any("@example.com" in t for t in html) and all("logo@2x.png" in t for t in html),
          "emails are replaced, and logo@2x.png is not an email")
    check(all("8.8." not in t for t in html) and any("192.168.0." in t for t in html),
          "public IPs are replaced, private ones stay")
    se = [r for r in rows if r["source"] == "se"]
    check(se and not any("stackoverflow.com" in (r.get("url") or "") for r in se), "Stack Exchange without Stack Overflow")
    so = [coder.tok.decode(tokens_of(r).tolist()) for r in rows if r["source"] == "so"]
    check(so and all("Answer:" in t and "not this one" not in t for t in so),
          "Stack Overflow: each question with its accepted answer only")
    check(any(r.get("author") for r in rows if r["source"] == "so") and any(r.get("repo") for r in rows if r["source"] == "py"),
          "the index keeps the attribution (authors, repositories)")
    vals = m1["val"]
    check(set(vals) == set(w) and all(v["tokens"] > 0 for v in vals.values()), "every source has its validation stream")
    vt = np.fromfile(outs["one"] / vals["web-a"]["name"], dtype="<u2")
    check(not ((vt == META_START) | (vt == META_END)).any(), "validation has no tags")
    dec = Decon.load(outs["one"] / "prep" / "decon.npz")
    check(dec.hit(coder.encode([PLANTED])[0]) >= 0 and dec.hit(coder.encode([TEMPLATE])[0]) < 0,
          "the decontamination set has the question and not the template")
    return outs


def test_reader(outs):
    import torch
    from ludwig_lm.shards import META_END, META_START, OutOfData, ShardReader, TokenStream, meta_mask, val_sets
    from ludwig_lm.data import load_tokenizer

    print("reader", flush=True)
    st = TokenStream(outs["s2"] / "manifest.json", [outs["s1"]])
    full = st.read(0, st.total)
    seq, micro, world = 64, 2, 2
    rs = [ShardReader(st, seq, micro, r, world) for r in range(world)]
    got = [[rs[r].next() for _ in range(5)] for r in range(world)]
    flat = []
    for b in range(5):
        for j in range(micro):
            for r in range(world):
                flat.append(got[r][b][j].numpy())
    check(np.array_equal(np.concatenate(flat), full[:5 * micro * world * seq]),
          "two ranks interleave windows and together cover the stream in order, across shard boundaries")
    state = rs[0].state_dict()
    again = ShardReader(st, seq, micro, 0, world, state=state)
    nxt = rs[0].next()
    check(torch.equal(again.next(), nxt) and torch.equal(rs[0].peek(state), nxt), "a reader restarted from its state continues exactly")
    bad = dict(state, build_id="other")
    try:
        ShardReader(st, seq, micro, 0, world, state=bad)
        check(False, "a state from another build is refused")
    except ValueError:
        check(True, "a state from another build is refused")
    try:
        ShardReader(st, seq, micro, 0, world, state={**state, "token": st.total - 10}).next()
        check(False, "the end of the attached shards raises OutOfData")
    except OutOfData:
        check(True, "the end of the attached shards raises OutOfData")
    S, E = META_START, META_END
    cases = [([1, S, 7, 8, E, 9], [0, 1, 1, 1, 1, 0]), ([7, E, 9, S, 8], [1, 1, 0, 1, 1]),
             ([9, 9, 9], [0, 0, 0]), ([S, 1, E, 2, S, 3, E, 4], [1, 1, 1, 0, 1, 1, 1, 0])]
    check(all(meta_mask(np.array(a)).tolist() == [bool(x) for x in m] for a, m in cases),
          "the metadata mask: spans, markers included, also when a window starts or ends inside one")
    strip = ShardReader(st, seq, micro, 0, 1, meta="strip")
    keep = ~meta_mask(full).numpy()
    expect = full[keep]
    taken = np.concatenate([strip.next().numpy().reshape(-1) for _ in range(6)])
    check(np.array_equal(taken, expect[:len(taken)]) and not np.isin(taken, [S, E]).any(),
          "strip mode: the same stream without the metadata spans")
    s2 = ShardReader(st, seq, micro, 0, 1, state=strip.state_dict(), meta="strip")
    check(np.array_equal(s2.next().numpy().reshape(-1), expect[len(taken):len(taken) + micro * seq]),
          "strip mode resumes exactly")
    tok = load_tokenizer(TOKENIZER)
    vs = val_sets(outs["one"] / "manifest.json", tok, seq, 2, 0, world)
    check(set(vs) == {"en", "es", "python"} and all(v[0].shape == (2, seq) and v[1] > 0 for v in vs.values()),
          "validation windows per domain, with their bytes")


def torchrun(tag, out, *args):
    cfg = WORK / "tiny_64k.json"
    c = json.loads((ROOT / "configs" / "tiny_afmoe.json").read_text())
    c.update(vocab_size=65536, pad_token_id=1, bos_token_id=0)
    cfg.write_text(json.dumps(c))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
           str(ROOT / "scripts" / "train.py"), "--config", str(cfg), "--tokenizer", str(TOKENIZER),
           "--shards", str(WORK / "s2" / "manifest.json"), "--shard-roots", str(WORK / "s1"),
           "--seq", "64", "--micro", "2", "--tokens-per-step", "512", "--steps", "8", "--warmup", "2",
           "--eval-every", "4", "--eval-windows", "2", "--log-every", "1", "--tag", tag, "--out", str(out), *args]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if os.name == "nt":
        env["USE_LIBUV"] = "0"
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:], sep="\n")
    return p.returncode, [json.loads(x) for x in open(out)] if Path(out).exists() else []


def test_training():
    print("training on shards (2 processes on CPU)", flush=True)
    out = WORK / "train.jsonl"
    out.unlink(missing_ok=True)
    ck = WORK / "ckpt"
    shutil.rmtree(ck, ignore_errors=True)
    ra, rows_a = torchrun("A", out, "--save-dir", str(ck), "--save-at", "4")
    rb, rows_b = torchrun("B", out, "--save-dir", str(ck / "b"), "--resume", str(ck / "step_000004"))
    check(ra == 0 and rb == 0, "both runs finish")
    sa = {r["step"]: r for r in rows_a if r["type"] == "step" and r["tag"] == "A"}
    sb = {r["step"]: r for r in rows_b if r["type"] == "step" and r["tag"] == "B"}
    check(set(sb) == {4, 5, 6, 7} and all(sa[k]["ce"] == sb[k]["ce"] and sa[k]["batch_sha"] == sb[k]["batch_sha"]
                                          for k in sb), "B's steps 4–7 equal A's bit for bit")
    chk = [r for r in rows_b if r["type"] == "resume_check" and r["tag"] == "B"]
    res = [r for r in rows_b if r["type"] == "resume" and r["tag"] == "B"]
    check(chk and chk[0]["batch_exact"] and res and res[0]["state_exact"], "state and first batch exact after resuming")
    ev = [r for r in rows_a if r["type"] == "eval" and r["tag"] == "A"]
    check(ev and all(set(r.get("val_bpb_by_domain", {})) == {"en", "es", "python"} for r in ev),
          "evaluations report bits per byte per domain")
    cfg = next(r for r in rows_a if r["type"] == "config" and r["tag"] == "A")
    check(cfg.get("shards") == 10 and cfg.get("meta") == "mask", "the config row names the build and the meta mode")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    special = make_hub()
    outs = test_builder(special)
    if outs:
        test_reader(outs)
        test_training()
    print("\nPASS" if not FAILS else f"\nFAIL ({len(FAILS)}):\n  " + "\n  ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
