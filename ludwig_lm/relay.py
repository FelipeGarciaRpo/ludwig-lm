"""The relay (D15): one trunk, trained in turns by the friends' Kaggle accounts.

**Hugging Face is the primary store**, a private model repository of the project's organization:

    lock.json                    who has the relay: account label, session, until when (a lease)
    checkpoints/index.json       the complete checkpoints, oldest first: the newest ``keep`` of each run (tag)
    checkpoints/<tag>/step_XXXXXX/  one full-state checkpoint (ludwig_lm.checkpoint), ~11.3 GB. The first
                                 trunk (tag ``tronco``) wrote ``checkpoints/step_XXXXXX/``; its entries keep
                                 that path
    logs/<session>.jsonl         each session's training log

Each run rotates only its own checkpoints, so the trunk restarted under a new tag (P30) leaves the last
checkpoints of the first one where they are.

Every write is one commit whose parent is the last head this session saw. If another session committed
in between (because it took the lock after it expired), the commit fails and this session stops: the lock
and its fence in one, with no server of our own (fencing tokens, where the commit is the token). A
checkpoint, the index that points to it, the rotation and the lock's renewal go in the same commit, so the
index never points to a half-uploaded checkpoint. ``super_squash_history`` at the end of each session frees
the storage of the checkpoints rotated out (deleting a file does not, while history keeps it).

**Kaggle is the backup:** two private datasets take turns (A, B). At the end of each session its last
checkpoint goes to the one that holds the older checkpoint, so a failed upload still leaves one complete.
Kaggle unpacks the uploaded ``checkpoint.tar`` into ``checkpoint/step_XXXXXX/`` (G0-0, 2026-11-17), and a
session reads that directory where Kaggle mounts it, without copying it.

A session resumes from the newest complete checkpoint among Hugging Face and both backups. Nothing here
depends on the model: the session runs ``scripts/train.py`` as a subprocess and moves its directories.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path

from .checkpoint import complete

LOCK, INDEX = "lock.json", "checkpoints/index.json"


class Conflict(RuntimeError):
    """Another session committed since this one's last head: it no longer has the relay."""


class Busy(RuntimeError):
    """The lock is held by a live session."""


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _said(text: str, n: int = 4) -> list[str]:
    """The last ``n`` lines of a command's output that are not progress bars (tqdm redraws them with \\r)."""
    lines = (s.strip() for s in (text or "").replace("\r", "\n").split("\n"))
    return [s for s in lines if s and "%|" not in s][-n:]


# ----------------------------------------------------------------------------- stores
class HFStore:
    """A Hugging Face repository as the relay's store (a private model repository)."""

    def __init__(self, repo: str, token: str | None = None):
        from huggingface_hub import HfApi
        self.repo, self.token, self.api = repo, token, HfApi(token=token)

    def head(self) -> str:
        return self.api.repo_info(self.repo, repo_type="model").sha

    def read(self, path: str, rev: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return Path(hf_hub_download(self.repo, path, revision=rev, token=self.token)).read_bytes()
        except EntryNotFoundError:
            return None

    def commit(self, adds: dict, deletes: list[str], parent: str, message: str) -> str:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        from huggingface_hub.errors import HfHubHTTPError
        ops = [CommitOperationAdd(p, src) for p, src in adds.items()]
        ops += [CommitOperationDelete(p, is_folder=p.endswith("/")) for p in deletes]
        try:
            info = self.api.create_commit(self.repo, ops, commit_message=message, parent_commit=parent,
                                          repo_type="model")
        except HfHubHTTPError as e:
            if self.head() != parent:
                raise Conflict(f"{self.repo} moved from {parent[:8]}") from e
            raise
        return info.oid

    def download(self, prefix: str, rev: str, dest) -> Path:
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo, revision=rev, allow_patterns=[f"{prefix}/*"], local_dir=dest, token=self.token)
        return Path(dest) / prefix

    def squash(self, message: str) -> str:
        self.api.super_squash_history(self.repo, branch="main", commit_message=message, repo_type="model")
        return self.head()


class LocalStore:
    """The same interface over a folder (tests, and a way to run the relay without Hugging Face): commits are
    trees of content-addressed files, and a commit whose parent is not the head fails."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not (self.root / "commits.json").exists():
            self._write([{"sha": "0" * 40, "parent": None, "tree": {}, "message": "empty"}])

    def _commits(self):
        return json.loads((self.root / "commits.json").read_text())

    def _write(self, commits):
        tmp = self.root / "commits.json.tmp"
        tmp.write_text(json.dumps(commits))
        os.replace(tmp, self.root / "commits.json")

    def _tree(self, rev):
        for c in self._commits():
            if c["sha"] == rev:
                return c["tree"]
        raise KeyError(rev)

    def head(self):
        return self._commits()[-1]["sha"]

    def read(self, path, rev):
        obj = self._tree(rev).get(path)
        return (self.root / "objects" / obj).read_bytes() if obj else None

    def commit(self, adds, deletes, parent, message):
        with self._lock:
            commits = self._commits()
            if commits[-1]["sha"] != parent:
                raise Conflict(f"moved from {parent[:8]} to {commits[-1]['sha'][:8]}")
            tree = dict(commits[-1]["tree"])
            for d in deletes:
                for p in [p for p in tree if p == d or (d.endswith("/") and p.startswith(d))]:
                    del tree[p]
            for p, src in adds.items():
                data = src if isinstance(src, bytes) else Path(src).read_bytes()
                obj = hashlib.sha256(data).hexdigest()
                if not (self.root / "objects" / obj).exists():
                    (self.root / "objects" / obj).write_bytes(data)
                tree[p] = obj
            sha = hashlib.sha1(f"{parent}{message}{json.dumps(tree, sort_keys=True)}{time.time()}".encode()).hexdigest()
            commits.append({"sha": sha, "parent": parent, "tree": tree, "message": message})
            self._write(commits)
            return sha

    def download(self, prefix, rev, dest):
        out = Path(dest) / prefix
        for p, obj in self._tree(rev).items():
            if p.startswith(prefix + "/"):
                (Path(dest) / p).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root / "objects" / obj, Path(dest) / p)
        return out

    def squash(self, message):
        with self._lock:
            last = self._commits()[-1]
            self._write([{"sha": last["sha"], "parent": None, "tree": last["tree"], "message": message}])
            keep = set(last["tree"].values())
            for f in (self.root / "objects").iterdir():
                if f.name not in keep:
                    f.unlink()
            return last["sha"]

    def history(self):
        return len(self._commits())

    def stored_bytes(self):
        return sum(f.stat().st_size for f in (self.root / "objects").iterdir())


# ----------------------------------------------------------------------------- the lock
class Lock:
    """A lease in ``lock.json``: whoever wrote it last, with ``expires`` in the future, has the relay."""

    def __init__(self, store, label: str, session: str, lease_s: float = 7200, clock=time.time):
        self.store, self.label, self.session, self.lease_s, self.clock = store, label, session, lease_s, clock

    def holder(self, rev: str) -> dict | None:
        raw = self.store.read(LOCK, rev)
        return json.loads(raw) if raw else None

    def body(self, since=None) -> bytes:
        now = self.clock()
        return json.dumps({"holder": self.label, "session": self.session, "since": since or now, "renewed": now,
                           "expires": now + self.lease_s}, indent=1).encode()

    def acquire(self, tries: int = 3) -> str:
        for _ in range(tries):
            head = self.store.head()
            cur = self.holder(head)
            if cur and cur["expires"] > self.clock() and cur["session"] != self.session:
                left = (cur["expires"] - self.clock()) / 60
                raise Busy(f"{cur['holder']} has the relay for {left:.0f} more minutes (session {cur['session'][:8]})")
            try:
                return self.store.commit({LOCK: self.body()}, [], head, f"lock: {self.label} takes the relay")
            except Conflict:
                continue
        raise Busy("another session took the lock at the same time")

    def renew(self, head: str) -> str:
        return self.store.commit({LOCK: self.body(self.holder(head)["since"])}, [], head, f"lock: {self.label} renews")

    def release(self, head: str) -> str:
        cur = self.holder(head) or {}
        body = dict(cur, expires=self.clock(), released=True)
        return self.store.commit({LOCK: json.dumps(body, indent=1).encode()}, [], head, f"lock: {self.label} releases")


# ----------------------------------------------------------------------------- the backup
class KaggleBackup:
    """Two private Kaggle datasets that take turns. Each session reads them as attached inputs (Kaggle mounts
    them under /kaggle/input) and writes one with the ``kaggle`` command line (the account's own token).

    A backup is uploaded as ``checkpoint.tar`` (the checkpoint's directory, uncompressed: DCP's ``.metadata``
    and the rest exactly as written) plus ``checkpoint.json`` (its meta.json), so the step is known without
    reading 11 GB. Kaggle unpacks the tar: the mounted dataset holds ``checkpoint.json`` and
    ``checkpoint/step_XXXXXX/``, hidden ``.metadata`` included, and no ``.tar``."""

    def __init__(self, slugs: list[str], roots=("/kaggle/input",), tag: str = "tronco", run=subprocess.run):
        if len(slugs) != 2:
            raise ValueError("the backup is two datasets that take turns")
        self.slugs, self.roots, self.tag, self.run = slugs, [Path(r) for r in roots], tag, run

    def copies(self) -> dict[str, tuple[int, Path]]:
        """The newest checkpoint of this run in each attached backup: {slug: (step, path)}. The path is the
        directory Kaggle unpacked, or ``checkpoint.tar`` if a copy is still packed."""
        out = {}
        for slug in self.slugs:
            name = slug.split("/")[-1]
            for r in self.roots:
                for m in list(r.glob(f"**/{name}/checkpoint.json")) + list(r.glob(f"**/{name}/**/checkpoint.json")):
                    meta = json.loads(m.read_text())
                    if meta.get("tag") != self.tag or meta["step"] <= out.get(slug, (-1, None))[0]:
                        continue
                    tar = m.parent / "checkpoint.tar"
                    found = self.unpacked(m.parent, meta["step"]) or (tar if tar.exists() else None)
                    if found is not None:
                        out[slug] = (meta["step"], found)
        return out

    @staticmethod
    def unpacked(root: Path, step: int) -> Path | None:
        """The checkpoint directory Kaggle unpacked next to ``checkpoint.json``, if it is complete: its meta.json
        is for ``step`` and DCP's ``.metadata`` is there (without it the checkpoint cannot be loaded)."""
        for d in complete(root / "checkpoint"):
            if (d / ".metadata").exists() and json.loads((d / "meta.json").read_text()).get("step") == step:
                return d
        return None

    @staticmethod
    def extract(tar_path: Path, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as t:
            names = t.getnames()
            t.extractall(dest, filter="data")
        top = names[0].split("/")[0]
        return dest / top

    def target(self) -> str:
        """The dataset to write: the one without a checkpoint, else the one with the older."""
        have = self.copies()
        missing = [s for s in self.slugs if s not in have]
        return missing[0] if missing else min(self.slugs, key=lambda s: have[s][0])

    def upload(self, ckpt_dir: Path, slug: str, message: str, stage: Path) -> None:
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        with tarfile.open(stage / "checkpoint.tar", "w") as t:
            t.add(ckpt_dir, arcname=ckpt_dir.name)
        shutil.copy(ckpt_dir / "meta.json", stage / "checkpoint.json")
        (stage / "dataset-metadata.json").write_text(json.dumps({"id": slug, "title": slug.split("/")[-1],
                                                                  "licenses": [{"name": "other"}]}))
        p = self.run(["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "-d"],
                     capture_output=True, text=True)
        if p.returncode != 0:                        # the cause, not the progress bars that hid it in turns 5-7
            said = _said(p.stdout) + _said(p.stderr)
            raise RuntimeError(f"kaggle datasets version {slug} (exit {p.returncode}): " + " | ".join(said)[-500:])


def train_command(args: dict, resume: str | None, save_dir: str, nproc: int = 2, python: str = "python") -> list[str]:
    """torchrun + scripts/train.py with ``args`` (the ``args`` of configs/tronco_v1.json plus the session's own)."""
    a = dict(args, save_dir=save_dir)
    if resume:
        a["resume"] = resume
    cmd = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc), "scripts/train.py"]
    for k, v in a.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd += [flag, *map(str, v)]
        else:
            cmd += [flag, str(v)]
    return cmd


# ----------------------------------------------------------------------------- one session
class Session:
    """One turn of the relay: lock, resume from the newest checkpoint, train, upload while training, back up,
    squash, release. ``train_cmd(resume, save_dir)`` gives the command that runs scripts/train.py."""

    def __init__(self, store, backup: KaggleBackup | None, label: str, tag: str, work, train_cmd, log_path,
                 lease_s: float = 7200, renew_s: float = 1800, poll_s: float = 60, keep: int = 2, squash: bool = True,
                 no_lock: bool = False, clock=time.time, say=print):
        self.store, self.backup, self.label, self.tag = store, backup, label, tag
        self.work, self.train_cmd, self.log_path = Path(work), train_cmd, Path(log_path)
        self.renew_s, self.poll_s, self.keep, self.squash_on, self.no_lock = renew_s, poll_s, keep, squash, no_lock
        self.clock, self.say = clock, say
        self.session = uuid.uuid4().hex
        self.lock = Lock(store, label, self.session, lease_s, clock) if store is not None else None
        self.head, self.uploaded, self.events = None, -1, []

    # ---- where to resume from
    def index(self) -> list[dict]:
        raw = self.store.read(INDEX, self.head) if (self.store is not None and self.head) else None
        return json.loads(raw)["checkpoints"] if raw else []

    def newest(self):
        """("hf" | "kaggle" | None, step, where): the newest complete checkpoint of this run."""
        best = (None, -1, None)
        if not self.no_lock:
            idx = [c for c in self.index() if c.get("tag") == self.tag]
            if idx:
                best = ("hf", idx[-1]["step"], idx[-1]["path"])
        if self.backup is not None:
            for slug, (step, path) in self.backup.copies().items():
                if step >= best[1]:                               # a tie goes to the copy already on disk
                    best = ("kaggle", step, path)
        return best

    def fetch(self, where, step, ref) -> Path | None:
        if where is None:
            return None
        if where == "kaggle":
            if Path(ref).is_dir():                  # unpacked by Kaggle: train.py only reads it, where it is mounted
                self.event("in_place", step=step)
                return Path(ref)
            t = time.time()
            path = self.backup.extract(Path(ref), self.work / "from_kaggle")
            self.event("extract", step=step, seconds=round(time.time() - t, 1))
            return path
        t = time.time()
        path = self.store.download(ref, self.head, self.work / "from_hf")
        self.event("download", step=step, seconds=round(time.time() - t, 1))
        return path

    # ---- writes
    def event(self, kind, **kw):
        row = {"kind": kind, "time": round(self.clock(), 1), "by": self.label, "session": self.session[:8], **kw}
        self.events.append(row)
        self.say(f"[relevo] {kind} " + " ".join(f"{k}={v}" for k, v in kw.items()))

    def free(self, path: Path, what: str):
        """Delete a local checkpoint the turn no longer needs, so /tmp never holds more than two at once
        (~23 GiB on the trunk; Kaggle gives ~73 GB of disk in all)."""
        if path.exists():
            gib = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
            shutil.rmtree(path, ignore_errors=True)
            self.event("free", what=what, GiB=round(gib, 2))

    def upload(self, ckpt: Path):
        meta = json.loads((ckpt / "meta.json").read_text())
        step, t, tag = meta["step"], time.time(), meta.get("tag") or self.tag
        idx = [c for c in self.index() if not (c.get("tag") == tag and c["step"] == step)]
        path = f"checkpoints/{tag}/{ckpt.name}"
        entry = {"step": step, "tokens": meta["tokens"], "path": path, "tag": tag,
                 "reason": meta.get("reason"), "by": self.label, "session": self.session[:8], "uploaded": self.clock(),
                 "files": {f.name: f.stat().st_size for f in ckpt.iterdir()}}
        idx.append(entry)
        drop = [c for c in idx if c.get("tag") == tag][:-self.keep]       # only this run's: other runs stay
        idx = [c for c in idx if c not in drop]
        adds = {f"{path}/{f.name}": str(f) for f in ckpt.iterdir()}
        adds[INDEX] = json.dumps({"checkpoints": idx}, indent=1).encode()
        if self.log_path.exists():
            adds[f"logs/{time.strftime('%Y%m%d', time.gmtime(self.clock()))}-{_slug(self.label)}-{self.session[:8]}.jsonl"] = \
                self.log_path.read_bytes()
        if not self.no_lock:
            adds[LOCK] = self.lock.body(self.lock.holder(self.head)["since"])
        self.head = self.store.commit(adds, [c["path"] + "/" for c in drop], self.head,
                                      f"checkpoint step {step} by {self.label}")
        self.uploaded = step
        self.last_renew = self.clock()
        gib = sum(entry["files"].values()) / 2**30
        self.event("upload", step=step, GiB=round(gib, 2), seconds=round(time.time() - t, 1),
                   kept=[c["step"] for c in idx if c.get("tag") == tag])

    # ---- the turn
    def run(self) -> dict:
        self.work.mkdir(parents=True, exist_ok=True)
        save_dir = self.work / "ckpt"
        if not self.no_lock:
            self.head = self.lock.acquire()
            self.last_renew = self.clock()
            self.event("lock", until=round(self.clock() + self.lock.lease_s))
        where, step, ref = self.newest()
        self.uploaded = max(self.uploaded, step)           # never upload again what this turn starts from
        self.event("resume_from", where=where, step=step)
        start = self.fetch(where, step, ref)
        local = self.work / ("from_kaggle" if where == "kaggle" else "from_hf")
        start_dir = local if start is not None and local in start.parents else None   # a copy this turn may delete
        cmd = self.train_cmd(str(start) if start else None, str(save_dir))
        proc = subprocess.Popen(cmd, start_new_session=True)
        lost = None
        try:
            while True:
                code = proc.poll()
                done = complete(save_dir)
                if start_dir is not None and done and json.loads((done[-1] / "meta.json").read_text())["step"] > step:
                    self.free(start_dir, "start")                   # train.py read it long before its first save
                    start_dir = None
                if not self.no_lock:
                    try:
                        if done and json.loads((done[-1] / "meta.json").read_text())["step"] > self.uploaded:
                            self.upload(done[-1])
                        elif self.clock() - self.last_renew > self.renew_s:
                            self.head = self.lock.renew(self.head)
                            self.last_renew = self.clock()
                    except Conflict as e:
                        lost = str(e)
                        self.event("lock_lost", error=lost)
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait()
                        break
                    except (OSError, RuntimeError) as e:            # Hugging Face down for a while: keep training
                        self.event("upload_failed", error=f"{type(e).__name__}: {str(e)[:200]}")
                if code is not None:
                    break
                time.sleep(self.poll_s)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
        code = proc.returncode
        summary = {"session": self.session, "by": self.label, "train_exit": code, "lock_lost": lost,
                   "resumed_from": {"where": where, "step": step}}
        done = complete(save_dir)
        last = done[-1] if done else None
        if lost is None and last is not None:
            if not self.no_lock and json.loads((last / "meta.json").read_text())["step"] > self.uploaded:
                try:
                    self.upload(last)
                except Exception as e:  # noqa: BLE001 - the Kaggle backup below still runs
                    self.event("upload_failed", error=f"{type(e).__name__}: {str(e)[:200]}")
            if self.backup is not None:
                for p in sorted(save_dir.glob("step_*")):           # room for the tar: only the last stays
                    if p != last:
                        self.free(p, p.name)
                if start_dir is not None:
                    self.free(start_dir, "start")
                slug = self.backup.target()
                t = time.time()
                try:
                    self.backup.upload(last, slug, f"{self.tag} step {last.name[5:]} by {self.label}", self.work / "backup")
                    self.event("backup", dataset=slug, step=int(last.name[5:]), seconds=round(time.time() - t, 1))
                except Exception as e:  # noqa: BLE001
                    self.event("backup_failed", error=f"{type(e).__name__}: {str(e)[:600]}")
        if not self.no_lock and lost is None:
            if self.squash_on:
                try:
                    self.head = self.store.squash(f"relay: history squashed after {self.label}'s session")
                    self.event("squash")
                except Exception as e:  # noqa: BLE001 - storage is freed next time
                    self.event("squash_failed", error=f"{type(e).__name__}: {str(e)[:200]}")
            try:
                self.head = self.lock.release(self.head)
                self.event("release")
            except Exception as e:  # noqa: BLE001 - the lease expires on its own
                self.event("release_failed", error=f"{type(e).__name__}: {str(e)[:200]}")
        summary.update(uploaded_step=self.uploaded, last_local=last.name if last else None, events=self.events)
        return summary
