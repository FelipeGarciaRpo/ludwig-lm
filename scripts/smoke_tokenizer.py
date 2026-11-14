"""T0.1b: the final 64K tokenizer in llama.cpp, in Ludwig 2's real format (AFMoE).

  T1  pre-tokenizer: llama.cpp knows the tokenizer's hash, or its split regex is one llama.cpp implements
  T2  a vocab-only AFMoE GGUF tokenizes like Hugging Face, string by string: prose in English and Spanish,
      code, math, and the chat, reasoning, tool-call and FIM markup
  T3  what llama.cpp makes of the GGUF: the pre-tokenizer, 65,536 tokens, the chat template, which added
      tokens are user-defined (<think> and the tool tags), and the special ids it detects on load
      (EOS, end of turn, padding, FIM)

Chat mode only: llama.cpp serves the model, it does not prepare training data. Data mode (no control id
from text) is tested in scripts/test_train_cpu.py.

CPU only; nearly all the time is compiling llama-tokenize. Writes results/t0_1b_tokenizer.json.

    python scripts/smoke_tokenizer.py [--llama-cpp DIR] [--tokenizer tokenizer/]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_format as sf  # noqa: E402  (test strings, regex, hash and llama.cpp helpers)

EXTRA = [
    "El niño comió 3½ arepas a las 7:45 p. m.; ¿cuántas quedan? «Ninguna» —dijo— y añadió: ¡qué pena!",
    "<|im_start|>system\nEres Ludwig.<|im_end|>\n<|im_start|>user\n¿Cuánto es 17 × 23?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n17 × 23 = 391.\n</think>\n391<|im_end|>\n",
    '<tool_call>\n{"name": "calc", "arguments": {"expr": "2**10"}}\n</tool_call>\n'
    "<tool_response>\n1024\n</tool_response>",
    "<|fim_prefix|>def area(r):\n    <|fim_suffix|>\n    return a<|fim_middle|>a = 3.14159 * r ** 2",
    "<|repo_name|>ludwig-lm<|file_sep|>README.md\n# Ludwig<|endoftext|><|pad|><|reserved_0|>",
    "const sum = (a: number, b: number): number => a + b;\nconsole.log(`total: ${sum(1, 2)}`);\n",
    "Sea $f(x)=\\frac{x^2-1}{x+1}$; entonces $\\lim_{x\\to -1} f(x) = -2$ y 1,234,567.89 ≈ 1.23e6.",
    "\n\n\n    \t\t  trailing   \n",
]
# what llama.cpp must detect on load, by its own log (print_info): required, then nice to have.
# BOS = EOS as in Qwen: never added (add_bos is false), but with no BOS id in the GGUF llama.cpp falls back to
# id 11, which here is <|fim_middle|>, and a template or tool that asks for the BOS would get a FIM token.
SPECIAL_REQUIRED = {"BOS": 0, "EOS": 0, "EOT": 3, "PAD": 1, "FIM PRE": 10, "FIM SUF": 12, "FIM MID": 11}
SPECIAL_OPTIONAL = {"FIM PAD": 13, "FIM REP": 14, "FIM SEP": 15}


def ensure_tokenize(llama_dir: Path, jobs: int) -> Path:
    bin_dir = llama_dir / "build" / "bin"
    if (bin_dir / "llama-tokenize").exists():
        return bin_dir
    if not (llama_dir / "CMakeLists.txt").exists():
        sf.log(f"[build] cloning llama.cpp into {llama_dir}")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(llama_dir)], check=True)
    sf.log("[build] compiling llama-tokenize (5-15 min on CPU)")
    subprocess.run(["cmake", "-S", str(llama_dir), "-B", str(llama_dir / "build"), "-DCMAKE_BUILD_TYPE=Release",
                    "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(llama_dir / "build"), "-j", str(jobs), "--target", "llama-tokenize"], check=True)
    return bin_dir


def gguf_view(llama_dir: Path, path: Path) -> dict:
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(str(path))

    def get(key):
        f = reader.fields.get(key)
        return None if f is None else f.contents()

    types = get("tokenizer.ggml.token_type") or []
    return {"architecture": get("general.architecture"), "model": get("tokenizer.ggml.model"),
            "pre": get("tokenizer.ggml.pre"), "n_tokens": len(get("tokenizer.ggml.tokens") or []),
            "eos": get("tokenizer.ggml.eos_token_id"), "pad": get("tokenizer.ggml.padding_token_id"),
            "bos": get("tokenizer.ggml.bos_token_id"), "add_bos": get("tokenizer.ggml.add_bos_token"),
            "chat_template": get("tokenizer.chat_template"),
            "user_defined": [i for i, t in enumerate(types) if t == gguf.TokenType.USER_DEFINED],
            "control": sum(t == gguf.TokenType.CONTROL for t in types)}


def llama_ids(bin_dir: Path, path: Path, text: str) -> list[int]:
    """llama.cpp's ids for a text. --no-escape: llama-tokenize would otherwise turn a literal backslash-t
    (LaTeX's \\to) into a tab."""
    r = subprocess.run([str(bin_dir / "llama-tokenize"), "-m", str(path), "-p", text, "--ids", "--no-escape",
                        "--log-disable"], capture_output=True, text=True)
    m = re.findall(r"\[[\d,\s]*\]", r.stdout)
    if not m:
        raise RuntimeError(f"llama-tokenize failed: {r.stdout[-500:]} {r.stderr[-500:]}")
    return json.loads(m[-1])


def loaded_specials(bin_dir: Path, path: Path) -> dict:
    """The special tokens llama.cpp reports when it loads the GGUF (print_info lines, verbosity 4)."""
    r = subprocess.run([str(bin_dir / "llama-tokenize"), "-m", str(path), "-p", "a", "--ids", "-lv", "4"],
                       capture_output=True, text=True)
    found: dict[str, list[int]] = {}
    for m in re.finditer(r"print_info: ([A-Z][A-Z ]*?) token\s*=\s*(\d+) ", r.stdout + r.stderr):
        found.setdefault(m.group(1).strip(), []).append(int(m.group(2)))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR", str(ROOT / "llama.cpp")))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "ludwig2_afmoe.json"))
    ap.add_argument("--work", default=str(ROOT / "work" / "t0_1b"))
    ap.add_argument("--out", default=str(ROOT / "results" / "t0_1b_tokenizer.json"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    t0 = time.time()
    tok_dir, llama_dir, work = Path(args.tokenizer), Path(args.llama_cpp), Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bin_dir = ensure_tokenize(llama_dir, args.jobs)
    report = {"llama_cpp": sf.llama_commit(llama_dir), "transformers": transformers.__version__,
              "tokenizers": tokenizers.__version__,
              "tokenizer_sha1": hashlib.sha1((tok_dir / "tokenizer.json").read_bytes()).hexdigest()}

    # ---- T1: pre-tokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    regex = sf.split_regex(tok_dir)
    pre = next((k for k, v in sf.KNOWN_PRE.items() if regex == v), None)
    h = sf.chkhsh(tok)
    known = sf.known_to_llama(llama_dir, h)
    report["T1_pretokenizer"] = {"pass": known or pre is not None, "chkhsh": h, "hash_known_to_llama_cpp": known,
                                 "split_regex": regex, "mapped_pre": pre}
    sf.log(f"[T1] hash known to llama.cpp: {known} | the split regex is llama.cpp's '{pre}'")

    # ---- the vocab-only GGUF, in the real format
    d = work / "hf"
    d.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(tok_dir / f, d / f)
    shutil.copy(args.config, d / "config.json")
    out = work / "ludwig64k_vocab.gguf"
    env = dict(os.environ, LLAMA_CPP_DIR=str(llama_dir))
    if not known and pre:
        env["LUDWIG_TOKENIZER_PRE"] = pre
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_gguf.py"), str(d), "--outfile", str(out),
                        "--outtype", "f32", "--vocab-only"], env=env, capture_output=True, text=True)
    if r.returncode != 0 or not report["T1_pretokenizer"]["pass"]:
        report["T2_parity"] = {"pass": False, "error": (r.stdout + r.stderr)[-3000:]}
        sf.log(f"[T2] conversion FAILED\n{(r.stdout + r.stderr)[-2000:]}")
    else:
        # ---- T2: same ids as Hugging Face (chat mode: control tokens in the text are tokens)
        strings = sf.TEST_STRINGS + EXTRA
        mismatches = []
        for s in strings:
            a, b = tok.encode(s, add_special_tokens=False), llama_ids(bin_dir, out, s)
            if a != b:
                mismatches.append({"text": s[:80], "hf": a[:48], "llama": b[:48]})
        report["T2_parity"] = {"pass": not mismatches, "n_strings": len(strings), "mismatches": mismatches}
        sf.log(f"[T2] identical tokenization: {len(strings) - len(mismatches)}/{len(strings)} strings")

        # ---- T3: what llama.cpp makes of it
        view = gguf_view(llama_dir, out)
        template = json.loads((tok_dir / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template")
        specials = loaded_specials(bin_dir, out)
        wrong = {k: [v, specials.get(k)] for k, v in SPECIAL_REQUIRED.items() if specials.get(k) != [v]}
        eog = specials.get("EOG", [])
        problems = [k for k, ok in {
            "pre-tokenizer is not qwen2": view["pre"] == "qwen2",
            "token count is not 65,536": view["n_tokens"] == 65536,
            "EOS is not 0 or padding is not 1": view["eos"] == 0 and view["pad"] == 1,
            "a BOS is added": view["add_bos"] is not True,
            "chat template differs": view["chat_template"] == template,
            "user-defined tokens are not ids 4-9": view["user_defined"] == list(range(4, 10)),
            "llama.cpp does not stop at <|endoftext|> and <|im_end|>": 0 in eog and 3 in eog,
        }.items() if not ok]
        report["T3_llama_cpp_view"] = {
            "pass": not problems and not wrong, "problems": problems, "wrong_special_ids": wrong,
            "gguf": {k: v for k, v in view.items() if k != "chat_template"},
            "detected_on_load": specials,
            "optional_not_detected": {k: [v, specials.get(k)] for k, v in SPECIAL_OPTIONAL.items()
                                      if specials.get(k) != [v]}}
        sf.log(f"[T3] llama.cpp's view: {'as expected' if report['T3_llama_cpp_view']['pass'] else 'PROBLEMS'} "
               f"| detected on load: {specials}")

    checks = {k: v for k, v in report.items() if isinstance(v, dict) and "pass" in v}
    report["all_pass"] = len(checks) == 3 and all(v["pass"] for v in checks.values())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sf.log("\n=== T0.1b summary ===")
    for k, v in checks.items():
        sf.log(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    sf.log(f"  -> {'ALL PASS' if report['all_pass'] else 'SOME CHECKS FAILED'} ({report['minutes']} min). Report: {args.out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
