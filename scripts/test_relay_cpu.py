#!/usr/bin/env python3
"""CPU test of the relay (ludwig_lm/relay.py): the lock, its fence, the store, the Kaggle backup, and a whole
relay of three sessions on the tiny model. No network: Hugging Face is a LocalStore and Kaggle a folder.

    python scripts/test_relay_cpu.py            # ~2–4 min; prints PASS or what failed

1. Lock: free → taken; a second session is refused while the lease lives; once it expires another
   session takes it, and the first one's next commit fails (the fence); releasing frees it.
2. Store: a commit on an old head fails; squashing keeps only the newest tree and frees the rest.
3. Backup: two datasets take turns (the empty one first, then the older). The fake Kaggle does what Kaggle
   does: it unpacks checkpoint.tar into checkpoint/step_XXXXXX/ and leaves it read-only, and the copy found
   is that directory, .metadata included. A copy still packed extracts to the same files, and an unpacked
   copy without .metadata does not count.
4. Relay on the tiny model (2 processes on CPU, T1 shards of scripts/test_shards_cpu.py's small Hub):
   R  12 steps without stopping (the reference);
   1  "cuenta 1": from scratch, saves at step 3 (uploaded while it trains) and stops at 6 like a time limit;
   2  "cuenta 2": resumes from the backup on disk (it ties with Hugging Face), reading it where it is
      unpacked, saves at 9, ends at 12;
   3  "cuenta 3": without the backup, downloads step 12 from the store and finds the run finished.
   Steps 6–11 must equal R's bit for bit, each resume must be exact, the store keeps the newest two
   checkpoints, and after the squash it holds nothing else.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
WORK = ROOT / "work" / "test_relay_cpu"
FAILS: list[str] = []

from ludwig_lm.relay import (INDEX, LOCK, Busy, Conflict, KaggleBackup, LocalStore, Lock, Session,  # noqa: E402
                             train_command)


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_lock():
    print("lock", flush=True)
    st = LocalStore(WORK / "store_lock")
    clk = Clock()
    a, b = Lock(st, "cuenta 1", "sa", lease_s=100, clock=clk), Lock(st, "cuenta 2", "sb", lease_s=100, clock=clk)
    ha = a.acquire()
    check(json.loads(st.read(LOCK, ha))["holder"] == "cuenta 1", "a free lock is taken")
    try:
        b.acquire()
        check(False, "a second session is refused while the lease lives")
    except Busy:
        check(True, "a second session is refused while the lease lives")
    clk.t += 50
    ha = a.renew(ha)
    clk.t += 120
    hb = b.acquire()
    check(json.loads(st.read(LOCK, hb))["holder"] == "cuenta 2", "once the lease expires, another session takes it")
    try:
        a.renew(ha)
        check(False, "the first session's next commit fails (the fence)")
    except Conflict:
        check(True, "the first session's next commit fails (the fence)")
    hb = b.release(hb)
    check(a.acquire() != hb, "a released lock can be taken at once")


def test_store():
    print("store", flush=True)
    st = LocalStore(WORK / "store_squash")
    h0 = st.head()
    h1 = st.commit({"a/x.bin": b"1" * 1000}, [], h0, "one")
    try:
        st.commit({"b": b"2"}, [], h0, "stale")
        check(False, "a commit on an old head fails")
    except Conflict:
        check(True, "a commit on an old head fails")
    h2 = st.commit({"c/y.bin": b"3" * 1000}, ["a/"], h1, "rotate")
    before = st.stored_bytes()
    h3 = st.squash("squash")
    check(st.history() == 1 and st.read("c/y.bin", h3) and st.read("a/x.bin", h3) is None and st.stored_bytes() < before,
          "squashing keeps the newest tree and frees what was rotated out")
    check(h2 == h3, "the head keeps its content after squashing")


def _rmtree(path: Path):
    """shutil.rmtree, also for the read-only files and folders the fake Kaggle leaves."""
    if path.exists():
        for p in [path, *path.rglob("*")]:
            os.chmod(p, 0o777 if p.is_dir() else 0o666)
        shutil.rmtree(path)


def fake_kaggle(root: Path, unpack: bool = True):
    """``kaggle datasets version`` as Kaggle does it (G0-0, 2026-11-17): the dataset gets checkpoint.json and the
    tar unpacked into checkpoint/, with no .tar left, read-only like /kaggle/input. unpack=False keeps the tar."""
    def run(cmd, capture_output=True, text=True):
        stage = Path(cmd[cmd.index("-p") + 1])
        slug = json.loads((stage / "dataset-metadata.json").read_text())["id"]
        owner, name = slug.split("/")
        dest = root / "datasets" / owner / name
        _rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copy(stage / "checkpoint.json", dest / "checkpoint.json")
        if unpack:
            with tarfile.open(stage / "checkpoint.tar") as t:
                t.extractall(dest / "checkpoint", filter="data")
            for p in dest.rglob("*"):
                os.chmod(p, 0o555 if p.is_dir() else 0o444)
        else:
            shutil.copy(stage / "checkpoint.tar", dest / "checkpoint.tar")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def test_backup():
    print("backup", flush=True)
    ck = WORK / "unit_ckpt"
    for step in (5, 8, 11):
        d = ck / f"step_{step:06d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".metadata").write_bytes(b"m" * 10)
        (d / "__0_0.distcp").write_bytes(bytes(range(256)) * 4)
        (d / "meta.json").write_text(json.dumps({"step": step, "tag": "t", "tokens": step * 10}))

    def same(a: Path, b: Path) -> bool:
        return sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir()) and all(
            (a / p.name).read_bytes() == p.read_bytes() for p in b.iterdir())

    slugs = ["dueno/respaldo-a", "dueno/respaldo-b"]
    for unpack in (True, False):
        how = "unpacked" if unpack else "packed"
        root = WORK / f"kaggle_unit_{how}"
        _rmtree(root)
        bk = KaggleBackup(slugs, roots=[root], tag="t", run=fake_kaggle(root, unpack))
        for step in (5, 8, 11):
            bk.upload(ck / f"step_{step:06d}", bk.target(), f"step {step}", WORK / "stage")
        have = bk.copies()
        check({s: v[0] for s, v in have.items()} == {"dueno/respaldo-a": 11, "dueno/respaldo-b": 8},
              f"{how}: two datasets take turns, empty first, then the one with the older checkpoint")
        path = have.get("dueno/respaldo-a", (None, None))[1]
        if path is None:
            check(False, f"{how}: the copy in dataset A is found")
        elif unpack:
            check(path.is_dir() and path.name == "step_000011" and same(path, ck / "step_000011"),
                  "unpacked as Kaggle does: the copy found is the checkpoint's directory, .metadata included")
        else:
            out = bk.extract(path, WORK / "extracted")
            check(path.name == "checkpoint.tar" and same(out, ck / "step_000011"),
                  "still packed: the tar extracts to the same files, .metadata included")
    gone = WORK / "kaggle_unit_unpacked" / "datasets" / "dueno" / "respaldo-a" / "checkpoint" / "step_000011" / ".metadata"
    os.chmod(gone.parent, 0o777)
    os.chmod(gone, 0o666)
    gone.unlink()
    bk = KaggleBackup(slugs, roots=[WORK / "kaggle_unit_unpacked"], tag="t")
    check({s: v[0] for s, v in bk.copies().items()} == {"dueno/respaldo-b": 8},
          "an unpacked copy without DCP's .metadata does not count (it could not be loaded)")


def make_shards():
    import test_shards_cpu as ts
    ts.WORK = WORK / "shards"
    ts.HUB = ts.WORK / "hub"
    ts.WORK.mkdir(parents=True, exist_ok=True)
    ts.make_hub()
    rec = ts.WORK / "recipe.json"
    rec.write_text(json.dumps(ts.recipe(), indent=1), encoding="utf-8")
    out = ts.WORK / "build"
    shutil.rmtree(out, ignore_errors=True)
    p = ts.run_build(["build", "--recipe", str(rec), "--tokenizer", str(ts.TOKENIZER), "--local-hub", str(ts.HUB),
                      "--out", str(out), "--shards", "10"])
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
    return out / "manifest.json"


def train_args(manifest, out, tag):
    cfg = WORK / "tiny_64k.json"
    c = json.loads((ROOT / "configs" / "tiny_afmoe.json").read_text())
    c.update(vocab_size=65536, pad_token_id=1, bos_token_id=0)
    cfg.write_text(json.dumps(c))
    return {"config": str(cfg), "tokenizer": str(ROOT / "tokenizer" / "tokenizer.json"), "shards": str(manifest),
            "seq": 64, "micro": 2, "tokens_per_step": 512, "steps": 12, "warmup": 2, "eval_every": 6,
            "eval_windows": 2, "log_every": 1, "save_final": True, "tag": tag, "out": str(out)}


def rows(path, tag, kind):
    return [r for r in map(json.loads, open(path)) if r["tag"] == tag and r["type"] == kind] if Path(path).exists() else []


def test_relay():
    print("relay on the tiny model (2 processes on CPU)", flush=True)
    manifest = make_shards()
    os.environ["OMP_NUM_THREADS"] = "1"
    log = WORK / "train.jsonl"
    log.unlink(missing_ok=True)
    ref = train_args(manifest, log, "R")
    p = subprocess.run(train_command(ref, None, str(WORK / "ref_ckpt"), python=sys.executable), cwd=ROOT,
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:])
    R = {r["step"]: r for r in rows(log, "R", "step")}
    check(p.returncode == 0 and len(R) == 12, "the reference runs 12 steps")

    store = LocalStore(WORK / "store")
    kroot = WORK / "kaggle"
    _rmtree(kroot)
    backup = KaggleBackup(["dueno/ludwig2-respaldo-a", "dueno/ludwig2-respaldo-b"], roots=[kroot], tag="tronco",
                          run=fake_kaggle(kroot))
    os.chdir(ROOT)

    def session(label, extra, use_backup=True):
        args = dict(train_args(manifest, log, "tronco"), **extra)
        cmd = lambda resume, save_dir: train_command(args, resume, save_dir, python=sys.executable)  # noqa: E731
        s = Session(store, backup if use_backup else None, label, "tronco", WORK / _slug(label), cmd, log,
                    lease_s=3600, renew_s=3600, poll_s=0.5)
        return s.run()

    from ludwig_lm.relay import _slug
    s1 = session("cuenta 1", {"save_at": [3], "pause_at_step": 6})
    ev1 = [e["kind"] for e in s1["events"]]
    check(s1["train_exit"] == 0 and ev1.count("upload") == 2 and "backup" in ev1 and "release" in ev1,
          "session 1: two uploads (step 3 while training, step 6 at the end), backup, squash, release")
    idx = json.loads(store.read(INDEX, store.head()))["checkpoints"]
    check([c["step"] for c in idx] == [3, 6], "the store's index lists steps 3 and 6")
    s2 = session("cuenta 2", {"save_at": [9]})
    check(s2["resumed_from"] == {"where": "kaggle", "step": 6}, "session 2 resumes from the backup on disk (a tie with the store)")
    check(any(e["kind"] == "in_place" for e in s2["events"]),
          "session 2 reads the backup where it is unpacked (read-only), without copying it")
    w2 = WORK / _slug("cuenta 2")
    freed = [e["what"] for e in s2["events"] if e["kind"] == "free"]
    check(not (w2 / "from_kaggle").exists() and [p.name for p in (w2 / "ckpt").glob("step_*")] == ["step_000012"]
          and "start" not in freed and "step_000009" in freed,
          "session 2 frees the disk: step 9 is gone before the backup, step 12 stays, and it copied nothing to start from")
    tr = {r["step"]: r for r in rows(log, "tronco", "step")}
    check(all(k in tr and tr[k]["ce"] == R[k]["ce"] and tr[k]["batch_sha"] == R[k]["batch_sha"] for k in range(12)),
          "steps 0–11 across two accounts equal the uninterrupted run bit for bit")
    res = rows(log, "tronco", "resume")
    chk = rows(log, "tronco", "resume_check")
    check(len(res) == 1 and res[0]["state_exact"] and chk and chk[0]["batch_exact"], "the resume is exact: state and first batch")
    idx = json.loads(store.read(INDEX, store.head()))["checkpoints"]
    check([c["step"] for c in idx] == [9, 12] and [c["by"] for c in idx] == ["cuenta 2", "cuenta 2"],
          "the store keeps the newest two checkpoints (9 and 12)")
    files = [p for p in json.loads((WORK / "store" / "commits.json").read_text())[-1]["tree"]]
    check(not any("step_000003" in f or "step_000006" in f for f in files) and store.history() == 2,
          "after the squash only the newest tree remains (and the release on top)")
    have = backup.copies()
    check({s: v[0] for s, v in have.items()} == {"dueno/ludwig2-respaldo-a": 6, "dueno/ludwig2-respaldo-b": 12},
          "the backups took turns: step 6 in A, step 12 in B")
    s3 = session("cuenta 3", {}, use_backup=False)
    check(s3["resumed_from"] == {"where": "hf", "step": 12} and s3["train_exit"] == 0,
          "session 3 downloads step 12 from the store and finds the run finished")
    lock = json.loads(store.read(LOCK, store.head()))
    check(lock["holder"] == "cuenta 3" and lock.get("released"), "the last session released the lock")
    logs = [f for f in json.loads((WORK / "store" / "commits.json").read_text())[-1]["tree"] if f.startswith("logs/")]
    check(logs and all("cuenta" in f for f in logs), "each session's log is in the store, named by its account label")


def main():
    _rmtree(WORK)
    WORK.mkdir(parents=True)
    test_lock()
    test_store()
    test_backup()
    test_relay()
    print("\nPASS" if not FAILS else f"\nFAIL ({len(FAILS)}):\n  " + "\n  ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
