#!/usr/bin/env python3
"""T0.1 — format, MTP and tokenizer smoke test. CPU only, no GPU hours.

Checks, on a tiny model with exactly Ludwig 2's structure (gated full attention
in every layer, partial RoPE, MoE + gated shared expert, tied embeddings, MTP):

  A1  export -> reload with transformers gives identical logits (mtp.* ignored)
  A2  convert to GGUF (f32) with llama.cpp's converter, MTP included
  A3  llama.cpp next-token top-5 probabilities == transformers (tolerance 2e-3)
  A4  llama.cpp greedy continuation == transformers greedy (token by token)
  A5  speculative decoding with the MTP head (--spec-type draft-mtp) gives the
      same greedy tokens, and reports draft acceptance
  T1  tokenizer: split-regex -> llama.cpp pre-tokenizer, hash known or mapped
  T2  tokenizer: llama.cpp tokenization == Hugging Face on EN/ES/code/math text

Run with --tokenizer DIR to run T1-T2 on Ludwig's real 64K tokenizer too.
Writes a JSON report (default results/t0_1.json). Exit code 0 only if all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ludwig_lm.model import LossWeights, LudwigLM, load_config  # noqa: E402
from ludwig_lm.muon import build_optimizers  # noqa: E402

# Original split regexes as they appear in tokenizer.json (llama.cpp src/llama-vocab.cpp comments).
KNOWN_PRE = {
    "llama-bpe": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "gpt-4o": r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}

# Same probe string llama.cpp hashes to identify a pre-tokenizer (conversion/base.py).
CHKTXT = ('\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 '
          '3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ '
          '------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, '
          '\'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL')

CORPUS = [
    "The quick brown fox jumps over the lazy dog. Language models learn to predict the next token.",
    "El rápido zorro marrón salta sobre el perro perezoso. ¿Dónde está la biblioteca? ¡Qué día tan bonito en Medellín!",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "Solve: 12 * 7 = 84. Then 84 / 4 = 21, so the answer is 21. Next: 3.14159 * 2 = 6.28318.",
    "import torch\nx = torch.randn(4, 8)\nprint(x.mean(dim=-1))  # average over the last axis\n",
    "Boltzmann wrote S = k log W. La entropía mide cuántos microestados son compatibles con un macroestado.",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>2+2=4</think>The answer is 4.<|im_end|>",
]
TEST_STRINGS = CORPUS[:6] + [
    "Hello   world\n\n\tTabs and    spaces 12345678 9.99 1e-5 ---===",
    "Ñandú, pingüino, acción, corazón; ¿por qué? ¡porque sí! 😀 🦙",
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n",
]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<think>", "</think>"]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- llama.cpp
def ensure_llama_cpp(llama_dir: Path, jobs: int) -> Path:
    bin_dir = llama_dir / "build" / "bin"
    if (bin_dir / "llama-server").exists() and (bin_dir / "llama-tokenize").exists():
        return bin_dir
    if not (llama_dir / "CMakeLists.txt").exists():
        log(f"[build] cloning llama.cpp into {llama_dir}")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(llama_dir)], check=True)
    log("[build] compiling llama-server and llama-tokenize (5-15 min on CPU)")
    subprocess.run(["cmake", "-S", str(llama_dir), "-B", str(llama_dir / "build"), "-DCMAKE_BUILD_TYPE=Release",
                    "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(llama_dir / "build"), "-j", str(jobs), "--target",
                    "llama-server", "llama-tokenize"], check=True)
    return bin_dir


def llama_commit(llama_dir: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(llama_dir), "log", "-1", "--format=%H %cd"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def convert(llama_dir: Path, hf_dir: Path, out: Path, outtype="f32", pre=None, vocab_only=False) -> subprocess.CompletedProcess:
    env = dict(os.environ, LLAMA_CPP_DIR=str(llama_dir))
    if pre:
        env["LUDWIG_TOKENIZER_PRE"] = pre
    cmd = [sys.executable, str(ROOT / "scripts" / "convert_gguf.py"), str(hf_dir), "--outfile", str(out), "--outtype", outtype]
    if vocab_only:
        cmd += ["--vocab-only", "--no-mtp"]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


class LlamaServer:
    def __init__(self, bin_dir: Path, model: Path, extra=()):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.log_path = Path(tempfile.mkstemp(suffix=".log")[1])
        cmd = [str(bin_dir / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port),
               "-c", "1024", "-np", "1", "--no-webui", *extra]
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(240):
            if self.proc.poll() is not None:
                text = self.log_path.read_text(errors="ignore")
                asserts = [l for l in text.splitlines() if "GGML_ASSERT" in l or "error" in l.lower()][:3]
                raise RuntimeError("llama-server exited: " + " | ".join(asserts) + "\n" + text[-2000:])
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("llama-server did not become healthy:\n" + self.log_path.read_text()[-3000:])

    def post(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def top_probs(entry, k):
    """Accept both server formats: top_logprobs (newer) or probs (older)."""
    items = entry.get("top_logprobs") or entry.get("probs") or []
    out = []
    for it in items[:k]:
        p = math.exp(it["logprob"]) if "logprob" in it else it.get("prob")
        out.append((int(it["id"]), float(p)))
    return out


# ----------------------------------------------------------------------------- tokenizer helpers
def train_tiny_tokenizer(out_dir: Path, vocab_size: int, regex: str):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(CORPUS * 50, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>",
                                   additional_special_tokens=SPECIALS[1:3] + SPECIALS[4:])
    fast.save_pretrained(out_dir)
    return fast


def split_regex(tokenizer_dir: Path):
    data = json.loads((tokenizer_dir / "tokenizer.json").read_text())

    def walk(node):
        if not isinstance(node, dict):
            return None
        if node.get("type") == "Split":
            pat = node.get("pattern", {})
            return pat.get("Regex") or pat.get("String")
        for child in node.get("pretokenizers", []) or []:
            r = walk(child)
            if r:
                return r
        return None

    return walk(data.get("pre_tokenizer") or {})


def chkhsh(tok) -> str:
    return hashlib.sha256(str(tok.encode(CHKTXT)).encode()).hexdigest()


def known_to_llama(llama_dir: Path, h: str) -> bool:
    for f in (llama_dir / "conversion").glob("*.py"):
        if h in f.read_text(errors="ignore"):
            return True
    return h in (llama_dir / "convert_hf_to_gguf.py").read_text(errors="ignore")


def llama_tokenize(bin_dir: Path, gguf: Path, text: str):
    r = subprocess.run([str(bin_dir / "llama-tokenize"), "-m", str(gguf), "-p", text, "--ids", "--log-disable"],
                       capture_output=True, text=True)
    m = re.findall(r"\[[\d,\s]*\]", r.stdout)
    if not m:
        raise RuntimeError(f"llama-tokenize failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    return json.loads(m[-1])


def check_tokenizer(name, tok_dir: Path, llama_dir: Path, bin_dir: Path, work: Path, base_cfg: dict, report: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    regex = split_regex(tok_dir)
    pre = next((k for k, v in KNOWN_PRE.items() if regex == v), None)
    h = chkhsh(tok)
    known = known_to_llama(llama_dir, h)
    t1_ok = known or pre is not None
    report[f"T1_{name}_pretokenizer"] = {"pass": t1_ok, "chkhsh": h, "hash_known_to_llama_cpp": known,
                                         "split_regex": regex, "mapped_pre": pre}
    log(f"[T1:{name}] hash known={known} | regex maps to: {pre}")
    if not t1_ok:
        log(f"[T1:{name}] FAIL: the split regex is not one llama.cpp implements. Use a supported regex (qwen2 or llama-bpe).")
        return
    # vocab-only GGUF with this tokenizer
    d = work / f"vocab_{name}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    cfg = dict(base_cfg, vocab_size=len(tok))
    (d / "config.json").write_text(json.dumps(cfg))
    tok.save_pretrained(d)
    out = work / f"vocab_{name}.gguf"
    r = convert(llama_dir, d, out, pre=None if known else pre, vocab_only=True)
    if r.returncode != 0:
        report[f"T2_{name}_parity"] = {"pass": False, "error": (r.stdout + r.stderr)[-2000:]}
        log(f"[T2:{name}] conversion FAILED\n{(r.stdout + r.stderr)[-1500:]}")
        return
    mismatches = []
    for s in TEST_STRINGS:
        a = tok.encode(s, add_special_tokens=False)
        b = llama_tokenize(bin_dir, out, s)
        if a != b:
            mismatches.append({"text": s[:60], "hf": a[:40], "llama": b[:40]})
    report[f"T2_{name}_parity"] = {"pass": not mismatches, "n_strings": len(TEST_STRINGS), "mismatches": mismatches}
    log(f"[T2:{name}] tokenization parity: {len(TEST_STRINGS) - len(mismatches)}/{len(TEST_STRINGS)} identical")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR", str(ROOT / "llama.cpp")))
    ap.add_argument("--tokenizer", default=None, help="dir with Ludwig's real tokenizer (tokenizer.json)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny.json"))
    ap.add_argument("--work", default=str(ROOT / "work" / "t0_1"))
    ap.add_argument("--out", default=None, help="default: results/t0_1_patched.json or results/t0_1_stock.json")
    ap.add_argument("--steps", type=int, default=80, help="quick training so logits and MTP are not random")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--patch", action="store_true",
                    help="apply patches/llama-cpp-hybrid-without-recurrent-layers.patch to llama.cpp and rebuild")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(ROOT / "results" / ("t0_1_patched.json" if args.patch else "t0_1_stock.json"))

    torch.manual_seed(0)
    t0 = time.time()
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    llama_dir = Path(args.llama_cpp)
    bin_dir = ensure_llama_cpp(llama_dir, args.jobs)
    patch = ROOT / "patches" / "llama-cpp-hybrid-without-recurrent-layers.patch"
    git_apply = lambda *a: subprocess.run(["git", "-C", str(llama_dir), "apply", *a, str(patch)],  # noqa: E731
                                          capture_output=True).returncode == 0
    patched = git_apply("--reverse", "--check")
    patch_note = None
    if args.patch and not patched and not git_apply("--check"):
        # upstream changed those lines (maybe fixed the bug): test this llama.cpp as it is
        patch_note = "patch does not apply to this llama.cpp; tested without it"
        log(f"[build] WARNING: {patch_note}")
    elif args.patch and not patched:
        log("[build] applying Ludwig's llama.cpp patch and rebuilding")
        subprocess.run(["git", "-C", str(llama_dir), "apply", str(patch)], check=True)
        subprocess.run(["cmake", "--build", str(llama_dir / "build"), "-j", str(args.jobs), "--target",
                        "llama-server", "llama-tokenize"], check=True)
        patched = True
    if not args.patch and patched:
        log("[build] reverting Ludwig's patch to test stock llama.cpp")
        subprocess.run(["git", "-C", str(llama_dir), "apply", "--reverse", str(patch)], check=True)
        subprocess.run(["cmake", "--build", str(llama_dir / "build"), "-j", str(args.jobs), "--target",
                        "llama-server", "llama-tokenize"], check=True)
        patched = False
    report = {"llama_cpp": llama_commit(llama_dir), "llama_cpp_patched": patched, "torch": torch.__version__}
    if patch_note:
        report["patch_note"] = patch_note
    import transformers
    report["transformers"] = transformers.__version__

    # ---- tiny tokenizer + tiny model
    tok_dir = work / "tiny_tokenizer"
    tok = train_tiny_tokenizer(tok_dir, 1024, KNOWN_PRE["qwen2"])
    base_cfg = json.loads(Path(args.config).read_text())
    cfg = load_config(args.config, vocab_size=len(tok))
    model = LudwigLM(cfg, with_mtp=True)
    ids = [i for s in CORPUS for i in tok.encode(s + "<|endoftext|>", add_special_tokens=False)]
    seq = 64
    data = torch.tensor((ids * (1 + 64 * seq // len(ids)))[: 64 * seq]).view(64, seq)
    muon, adamw = build_optimizers(model, lr=3e-3, n_heads=cfg.num_attention_heads, head_dim=cfg.head_dim)
    for step in range(args.steps):
        batch = data[torch.randint(0, data.size(0), (8,))]
        loss, met = model(batch, LossWeights(mtp=1.0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step(); adamw.step(); muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
    log(f"[train] {args.steps} steps: ce={float(met['ce']):.3f} mtp_ce={float(met['mtp_ce']):.3f}")
    model.eval()

    # ---- A1: export and reload
    hf_dir = work / "tiny_hf"
    model.save_hf(str(hf_dir), tokenizer=tok, dtype=torch.float32)
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    probe = data[:2, :32]
    with torch.no_grad():
        diff = (model.lm(input_ids=probe).logits - reloaded(input_ids=probe).logits).abs().max().item()
    report["A1_export_reload"] = {"pass": diff < 1e-5, "max_abs_diff": diff}
    log(f"[A1] reload max |diff| = {diff:.2e}")

    # ---- A2: GGUF conversion (with MTP)
    srv = None
    gguf = work / "tiny-f32.gguf"
    r = convert(llama_dir, hf_dir, gguf, pre="qwen2")
    ok = r.returncode == 0 and gguf.exists()
    report["A2_convert"] = {"pass": ok, "log_tail": (r.stdout + r.stderr)[-1500:]}
    log(f"[A2] convert: {'OK' if ok else 'FAILED'}")
    if not ok:
        log((r.stdout + r.stderr)[-3000:])
    else:
        prompts = [tok.encode(s, add_special_tokens=False)[:24] for s in CORPUS[:4]]
        n_new = 24
        banned = sorted({tok.convert_tokens_to_ids(t) for t in SPECIALS})
        bias = [[i, False] for i in banned]

        def hf_greedy(p):
            seqt = torch.tensor([p])
            with torch.no_grad():
                for _ in range(n_new):
                    logits = model.lm(input_ids=seqt).logits[0, -1]
                    logits[banned] = float("-inf")
                    nxt = logits.argmax()
                    seqt = torch.cat([seqt, nxt.view(1, 1)], dim=1)
            return seqt[0, len(p):].tolist()

        greedy_ref = [hf_greedy(p) for p in prompts]
        try:
            srv = LlamaServer(bin_dir, gguf, extra=["-t", str(args.jobs)])
        except Exception as e:
            srv = None
            msg = str(e)
            summary = msg.splitlines()[0][:400]          # "llama-server exited: <GGML_ASSERT line>"
            for k in ("A3_next_token_probs", "A4_greedy", "A5_mtp_speculative"):
                report[k] = {"pass": False, "error": summary, "log_tail": msg[-1500:]}
            log("[A3-A5] llama-server CRASHED loading the model (expected on stock llama.cpp for this format):")
            log("        " + summary)
    if not ok or srv is None:
        pass
    else:
        try:
            # A3: next-token distribution
            worst, id_mismatch = 0.0, 0
            for p in prompts:
                with torch.no_grad():
                    probs = model.lm(input_ids=torch.tensor([p])).logits[0, -1].float().softmax(-1)
                ref = torch.topk(probs, 5)
                res = srv.post("/completion", {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 5,
                                               "cache_prompt": False})
                got = top_probs(res["completion_probabilities"][0], 5)
                if [i for i, _ in got] != ref.indices.tolist():
                    id_mismatch += 1
                for (i, pr) in got:
                    worst = max(worst, abs(pr - probs[i].item()))
            report["A3_next_token_probs"] = {"pass": id_mismatch == 0 and worst < 2e-3, "max_abs_prob_diff": worst,
                                             "top5_order_mismatches": id_mismatch}
            log(f"[A3] top-5 order mismatches={id_mismatch}, max |p diff|={worst:.2e}")
            # A4: greedy continuation
            greedy_llama = []
            for p in prompts:
                res = srv.post("/completion", {"prompt": p, "n_predict": n_new, "temperature": 0, "logit_bias": bias,
                                               "return_tokens": True, "cache_prompt": False})
                greedy_llama.append(res["tokens"])
            same = sum(a == b for a, b in zip(greedy_ref, greedy_llama))
            report["A4_greedy"] = {"pass": same == len(prompts), "identical": f"{same}/{len(prompts)}"}
            log(f"[A4] greedy identical: {same}/{len(prompts)}")
        finally:
            srv.stop()

        # A5: speculative decoding with the MTP head
        try:
            spec = LlamaServer(bin_dir, gguf, extra=["-t", str(args.jobs), "--spec-type", "draft-mtp"])
            try:
                same, drafted, accepted = 0, 0, 0
                for p, ref in zip(prompts, greedy_ref):
                    res = spec.post("/completion", {"prompt": p, "n_predict": n_new, "temperature": 0, "logit_bias": bias,
                                                    "return_tokens": True, "cache_prompt": False})
                    same += res["tokens"] == ref
                    t = res.get("timings", {})
                    drafted += int(t.get("draft_n", 0))
                    accepted += int(t.get("draft_n_accepted", 0))
                acc = accepted / drafted if drafted else None
                report["A5_mtp_speculative"] = {"pass": same == len(prompts) and drafted > 0,
                                                "identical": f"{same}/{len(prompts)}", "drafted": drafted,
                                                "accepted": accepted, "acceptance": acc}
                log(f"[A5] MTP speculative: identical {same}/{len(prompts)}, drafted={drafted}, accepted={accepted}")
            finally:
                spec.stop()
        except Exception as e:  # keep the report even if the server refuses the flag
            report["A5_mtp_speculative"] = {"pass": False, "error": str(e)[-2000:]}
            log(f"[A5] FAILED: {str(e)[-800:]}")

    # ---- T1/T2: tokenizers
    check_tokenizer("tiny", tok_dir, llama_dir, bin_dir, work, base_cfg, report)
    if args.tokenizer:
        check_tokenizer("ludwig64k", Path(args.tokenizer), llama_dir, bin_dir, work, base_cfg, report)

    checks = {k: v for k, v in report.items() if isinstance(v, dict) and "pass" in v}
    report["all_pass"] = all(v["pass"] for v in checks.values())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("\n=== T0.1 summary ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    log(f"  -> {'ALL PASS' if report['all_pass'] else 'SOME CHECKS FAILED'} ({report['minutes']} min). Report: {args.out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
