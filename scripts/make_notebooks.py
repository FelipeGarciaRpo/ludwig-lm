"""Generate the Kaggle notebooks in notebooks/ from code, so they stay reviewable in diffs.

    python scripts/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
LLAMA_TESTED = "dd6b945be2188da875ab05a259ce94c6b2736e64"   # llama.cpp commit used for results/t0_1_*.json

SETUP = r'''import os, subprocess, sys, json
os.chdir("/kaggle/working")
if not os.path.exists("ludwig-lm"):
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "ludwig-lm"], check=True)
    else:  # el repo subido como dataset de Kaggle: se busca la carpeta que contiene ludwig_lm/
        import glob
        src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/ludwig_lm/__init__.py", recursive=True)[0]))
        subprocess.run(["cp", "-r", src, "ludwig-lm"], check=True)
os.chdir("/kaggle/working/ludwig-lm")
!pip install -q "transformers==5.22.2" safetensors sentencepiece protobuf
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| CPUs", os.cpu_count(),
      "| GPUs", torch.cuda.device_count())'''

RUN_HELPER = r'''import threading, time
RESULTS = "@RESULTS@"
KEEP = ("[model]", "  step", "[result]", "[ckpt]", "Error", "OutOfMemory", "out of memory", "Traceback")

def run(tag, timeout_min=25, **kw):
    """Lanza una medición con torchrun en las 2 GPUs, muestra el progreso y devuelve su fila de resultados."""
    args = " ".join(f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}") for k, v in kw.items())
    cmd = (f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 2 "
           f"@SCRIPT@ --tag {tag} --out {RESULTS} {args}")
    print(">>", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    killed = []
    def kill():  # torchrun runs each worker in its own session: kill by command line, TERM first, then KILL
        killed.append(True)
        subprocess.run(["pkill", "-TERM", "-f", "@SCRIPT@"])
        time.sleep(20)
        subprocess.run(["pkill", "-KILL", "-f", "@SCRIPT@"])
    timer = threading.Timer(timeout_min * 60, kill)
    timer.start()
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-30:]
        if any(k in line for k in KEEP):
            print(line, end="", flush=True)
    p.wait()
    timer.cancel()
    if p.returncode != 0:
        print(f"[{tag}] terminó con código {p.returncode}; últimas líneas:\n" + "".join(tail[-12:]))
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    row = next((r for r in reversed(rows) if r["tag"] == tag), None)
    if row is None:  # se cayó o se pasó del tiempo antes de escribir su fila: se anota igual para el resumen
        row = {"tag": tag, "status": "timeout" if killed else f"crash (código {p.returncode})", **kw}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
    return row'''

# ----------------------------------------------------------------------------- T0.1
t01 = nbf.v4.new_notebook()
t01.cells = [
    md("""# Ludwig 2 · T0.1 — Formato, MTP y tokenizador (CPU)

Prueba, **sin gastar horas de GPU**, que un modelo con la estructura exacta de Ludwig 2 (atención completa con compuerta en todas las capas, RoPE parcial, MoE + experto compartido, embeddings atados, cabeza MTP) se puede publicar:

| Prueba | Qué verifica |
|---|---|
| A1 | Exportar y recargar en transformers da logits idénticos |
| A2 | El conversor de llama.cpp genera el GGUF, con el MTP |
| A3 | llama.cpp y transformers dan las mismas probabilidades del siguiente token |
| A4 | Las continuaciones greedy son idénticas token a token |
| A5 | La decodificación especulativa con el MTP da exactamente lo mismo, y cuántos borradores acepta |
| T1–T2 | El tokenizador se reconoce en llama.cpp y tokeniza igual que Hugging Face |

**Se corre dos veces:** con llama.cpp sin cambios (se espera que A3–A5 fallen porque llama.cpp se cae con este formato) y con el parche de Ludwig (todo debe pasar).

**Kaggle:** acelerador *None* (CPU), Internet *On*. Tarda ~15–25 min, casi todo compilando llama.cpp."""),
    code(f'''REPO_URL = "https://github.com/FelipeGarciaRpo/ludwig-lm.git"   # o "" si subiste el repo como dataset
TOKENIZER_DIR = None   # p. ej. "/kaggle/input/ludwig-tokenizer-64k" (carpeta con tokenizer.json)
LLAMA_REF = "{LLAMA_TESTED}"   # commit con el que se obtuvieron los resultados del repo; "master" = el último'''),
    code(SETUP),
    md("## 1. Descargar y compilar llama.cpp (sin cambios)"),
    code(r'''LLAMA = "/kaggle/working/llama.cpp"
if not os.path.exists(LLAMA):
    !git init -q {LLAMA} && git -C {LLAMA} remote add origin https://github.com/ggml-org/llama.cpp
    !git -C {LLAMA} fetch -q --depth 1 origin {LLAMA_REF} && git -C {LLAMA} checkout -q FETCH_HEAD
!git -C {LLAMA} log -1 --format="llama.cpp %H (%cd)"
!cmake -S {LLAMA} -B {LLAMA}/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
!cmake --build {LLAMA}/build -j {os.cpu_count()} --target llama-server llama-tokenize 2>&1 | tail -2'''),
    md("## 2. Prueba con llama.cpp sin cambios (se espera que A3–A5 fallen)"),
    code(r'''tok = f"--tokenizer {TOKENIZER_DIR}" if TOKENIZER_DIR else ""
!python scripts/smoke_format.py --llama-cpp {LLAMA} --out results/t0_1_stock.json {tok} 2>&1 | grep -v -i warn | tail -25'''),
    md("## 3. Prueba con el parche de Ludwig (todo debe pasar)"),
    code(r'''!python scripts/smoke_format.py --llama-cpp {LLAMA} --patch --out results/t0_1_patched.json {tok} 2>&1 | grep -v -i warn | tail -25'''),
    md("## 4. Resumen"),
    code(r'''import pandas as pd
rows = []
for name in ["stock", "patched"]:
    r = json.load(open(f"results/t0_1_{name}.json"))
    for k, v in r.items():
        if isinstance(v, dict) and "pass" in v:
            rows.append({"llama.cpp": name, "prueba": k, "pasa": "✅" if v["pass"] else "❌",
                         "detalle": {kk: vv for kk, vv in v.items() if kk not in ("pass", "log_tail", "error", "mismatches")}})
    if r.get("patch_note"):
        print(f"{name}: {r['patch_note']}")
pd.set_option("display.max_colwidth", 120)
display(pd.DataFrame(rows))'''),
    md("""## Qué hacer con el resultado

- Descarga `results/t0_1_stock.json` y `results/t0_1_patched.json` (panel *Output*) y súbelos al repo: son la evidencia del diseño.
- Si corriste con `TOKENIZER_DIR`, mira `T1_ludwig64k_pretokenizer` y `T2_ludwig64k_parity`. Si T1 falla, el regex del tokenizador no es uno que llama.cpp implemente y hay que cambiarlo **antes** del tronco.
- Si con `LLAMA_REF = "master"` la prueba sin parche ya pasa, llama.cpp arregló el fallo por su cuenta y el parche sobra.
- El parche y el texto del PR para llama.cpp están en `patches/`."""),
]

# ----------------------------------------------------------------------------- T0.2
t02 = nbf.v4.new_notebook()
t02.cells = [
    md("""# Ludwig 2 · T0.2 — Velocidad y memoria en 2×T4

Mide con el **modelo real** (1.96B totales / 0.49B activos), FSDP2 entre las dos T4, FP16 con escalado de pérdida, Muon (momento bf16) + AdamW, pérdida por bloques y balanceo. Solo los datos son aleatorios.

| Bloque | Qué decide |
|---|---|
| A | El kernel de MoE más rápido en la T4: `eager`, `batched_mm` o `grouped_mm` |
| B | Si un micro-batch más grande sube los tokens/s sin pasarse de memoria |
| C–D | Cuánto cuestan 4K y 16K de contexto, y si 16K cabe |
| E | El plan B de 32 expertos |
| F | El costo de entrenar el MTP |
| G | Guardado y carga del checkpoint distribuido: tiempo, tamaño y restauración exacta |

**Kaggle:** acelerador *GPU T4 x2*, Internet *On*. Tarda ~45–75 min. Cada medición escribe una línea en `results/t0_2.jsonl`, así que si la sesión se corta no se pierde lo ya medido."""),
    code(r'''REPO_URL = "https://github.com/FelipeGarciaRpo/ludwig-lm.git"   # o "" si subiste el repo como dataset'''),
    code(SETUP + "\n!nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv\n!df -h /tmp /kaggle/working | tail -2"),
    code(RUN_HELPER.replace("@RESULTS@", "results/t0_2.jsonl").replace("@SCRIPT@", "scripts/bench_train.py")),
    md("""## A. Kernel de MoE (2K de contexto, micro-batch 4)

En la T4 (SM75), `grouped_mm` no tiene el kernel nativo de PyTorch (pide SM80+): transformers usa su respaldo, que ordena los tokens por experto y hace una multiplicación por experto. `batched_mm` y `eager` son las otras dos formas."""),
    code(r'''res = {}
for impl in ["grouped_mm", "batched_mm", "eager"]:
    res[impl] = run(f"A-{impl}", moe_impl=impl, seq=2048, micro=4, accum=2, steps=4)
ok = {k: v for k, v in res.items() if v and v["status"] == "ok"}
BEST = max(ok, key=lambda k: ok[k]["tokens_per_s"]) if ok else "batched_mm"
print("Kernel más rápido:", BEST)'''),
    md("## B–F. Micro-batch, contexto, plan B y MTP"),
    code(r'''run("B-micro8", moe_impl=BEST, seq=2048, micro=8, accum=1, steps=4)
run("C-seq4k", moe_impl=BEST, seq=4096, micro=2, accum=2, steps=4)
run("D-seq16k", moe_impl=BEST, seq=16384, micro=1, accum=1, steps=3)
run("E-32experts", moe_impl=BEST, experts=32, seq=2048, micro=4, accum=2, steps=4)
run("F-mtp", moe_impl=BEST, seq=2048, micro=4, accum=2, steps=4, mtp=1.0)'''),
    md("## G. Checkpoint distribuido: guardar y cargar"),
    code(r'''# ~12 GB: va a /tmp, no a /kaggle/working (que guarda la salida del notebook y tiene tope de ~20 GB)
run("G-ckpt", moe_impl=BEST, seq=2048, micro=4, accum=1, steps=3, ckpt_test=True, ckpt_dir="/tmp/ludwig_ckpt_test")
!rm -rf /tmp/ludwig_ckpt_test'''),
    md("## Resumen y recomendación"),
    code(r'''import pandas as pd
df = pd.DataFrame([json.loads(l) for l in open(RESULTS)])
cols = ["tag", "status", "experts", "seq", "micro", "accum", "moe_impl", "tokens_per_s", "step_s", "peak_reserved_GiB",
        "hours_for_25B", "loss_scale_skips", "load_max_over_mean", "ckpt_save_s", "ckpt_GiB", "ckpt_restore_exact"]
display(df[[c for c in cols if c in df.columns]])
ok40 = df[(df.status == "ok") & (df.experts == 40) & (df.seq == 2048)]
fits = ok40[ok40.peak_reserved_GiB <= 13.5]
if len(fits):
    b = fits.sort_values("tokens_per_s", ascending=False).iloc[0]
    print(f"Mejor configuración a 2K que cabe con margen: {b.tag} -> {b.tokens_per_s:,.0f} tokens/s, "
          f"pico {b.peak_reserved_GiB} GiB, ~{b.hours_for_25B:,.0f} h para 25B tokens")
elif len(ok40):
    print("Ninguna configuración de 40 expertos quedó bajo 13.5 GiB: mirar el plan B (fila E-32experts).")
else:
    print("Ninguna medición de 40 expertos a 2K terminó bien: revisar los errores de arriba.")'''),
    md("""## Qué hacer con el resultado

Descarga `results/t0_2.jsonl` y súbelo al repo. Con esos números se fijan el kernel de MoE, el micro-batch, si 16K cabe y si van 40 o 32 expertos. Después viene la compuerta G0: una prueba sostenida de ≥6 h con datos reales, con guardado y reanudación en otra cuenta."""),
]


# ----------------------------------------------------------------------------- T0.2 v2 (AFMoE)
t02v2 = nbf.v4.new_notebook()
t02v2.cells = [
    md("""# Ludwig 2 · T0.2 v2: el entrenamiento real en AFMoE, en 2×T4

Mide el paso de entrenamiento de verdad: el modelo de 1.906B parámetros totales y 0.491B activos en el formato AFMoE (`configs/ludwig2_afmoe.json`), FSDP2 entre las dos T4, FP16 con escalado de pérdida, Muon + AdamW, la pérdida por bloques con la cabeza atada, las pérdidas de balanceo y el bias de los expertos. Solo los datos son sintéticos.

La base lleva todas las piezas de velocidad. Cada medición de B a F quita o cambia **una sola** pieza, para ver cuánto vale:

| Medición | Qué cambia frente a la base | Qué decide |
|---|---|---|
| A | Nada: bmm con relleno, residuo en FP32, gradientes en FP16, prefetch, 8 secuencias de 2K por GPU | Tokens/s y memoria de referencia |
| B | Un matmul por experto (`grouped_mm`, que en la T4 cae a un loop) | Si el bmm con relleno vale la pena |
| C | Residuo en FP16 | Cuánto cuesta en velocidad y memoria mantenerlo en FP32 |
| D | Gradientes en FP32 entre las GPUs | Cuánto ahorra mandarlos en FP16 |
| E | Sin prefetch explícito de FSDP2 | Si el prefetch tapa la comunicación |
| F | Momento de Muon en la RAM de la CPU | Cuánta memoria libera y qué cuesta |
| G | 4 secuencias por GPU | El costo de partir el micro-lote, que el banco v1 dedujo pero no midió |
| H, I | 4K y 16K de contexto | Si caben y a qué velocidad |
| J | Guardar y cargar el checkpoint completo | Tiempo, tamaño y restauración exacta |
| K, L, M | 60 pasos de entrenamiento con el texto sintético del banco v1 | Balanceo de expertos: signo fijo (como el banco v1) contra SMEBU |

**Kaggle:** acelerador *GPU T4 x2*, Internet *On*. Tarda **~1–1.25 h**. Cada medición escribe su línea en `results/t0_2v2.jsonl` apenas termina: si la sesión se corta, lo medido no se pierde."""),
    code(r'''REPO_URL = "https://github.com/FelipeGarciaRpo/ludwig-lm.git"   # o "" si subiste el repo como dataset'''),
    code(SETUP + "\n!nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv\n!df -h /tmp /kaggle/working | tail -2"),
    code(RUN_HELPER.replace("@RESULTS@", "results/t0_2v2.jsonl").replace("@SCRIPT@", "scripts/bench_afmoe.py")),
    md("## A–I. Velocidad y memoria"),
    code(r'''ok = lambda r: bool(r) and r.get("status") == "ok"
BASE = dict(seq=2048, micro=8, accum=1, steps=4)
R = {}
R["A"] = run("A-base", **BASE)
if not ok(R["A"]):
    # 8 secuencias no caben: el resto mide con 4 (y 2 micro-pasos, los mismos 32K tokens por paso)
    print(">> 8 secuencias por GPU no caben; el resto usa 4")
    BASE = dict(BASE, micro=4, accum=2)
    R["A4"] = run("A-base-4-secuencias", **BASE)
R["B"] = run("B-loop", **BASE, moe_impl="grouped_mm")
R["C"] = run("C-residuo-fp16", **BASE, residual="fp16")
R["D"] = run("D-grad-fp32", **BASE, reduce="fp32")
R["E"] = run("E-sin-prefetch", **BASE, no_prefetch=True)
R["F"] = run("F-momento-cpu", **BASE, momentum_device="cpu")
if BASE["micro"] == 8:
    R["G"] = run("G-4-secuencias", **dict(BASE, micro=4, accum=2))
R["H"] = run("H-4k", **dict(BASE, seq=4096, micro=BASE["micro"] // 2))
R["I"] = run("I-16k", **dict(BASE, seq=16384, micro=1, steps=3))'''),
    md("## J. Checkpoint completo: guardar y cargar\n\nVa con el momento en la CPU si F liberó más de 1 GiB sin perder más de 3% de velocidad, porque así sería el entrenamiento real. El checkpoint (~12 GB) va a `/tmp`, no a `/kaggle/working`, que guarda la salida del notebook y tiene un tope de ~20 GB."),
    code(r'''base = next((R[k] for k in ("A", "A4") if ok(R.get(k))), None)
usar_cpu = (base is not None and ok(R["F"]) and base["peak_reserved_GiB"] - R["F"]["peak_reserved_GiB"] > 1.0
            and R["F"]["tokens_per_s"] >= 0.97 * base["tokens_per_s"])
print("Momento en la CPU para el checkpoint:", usar_cpu)
R["J"] = run("J-checkpoint", **dict(BASE, steps=2), ckpt_test=True, ckpt_dir="/tmp/ludwig_ckpt_test",
             momentum_device="cpu" if usar_cpu else "gpu")
!rm -rf /tmp/ludwig_ckpt_test'''),
    md("""## K–M. Balanceo de expertos en un entrenamiento corto

Son 60 pasos de 32K tokens (4 secuencias por GPU y 2 micro-pasos, como el banco v1) con el texto sintético del banco v1 (una cadena de Markov de 4,096 tokens, donde la segunda mitad de cada secuencia copia la primera). Allá, el bias con signo fijo de 0.001 dejó al experto más cargado en **8.4×** el promedio.

- **K:** el mismo signo fijo de 0.001, ahora en AFMoE, para comparar.
- **L, M:** SMEBU con pasos de 0.01 y 0.001."""),
    code(r'''TRAIN = dict(seq=2048, micro=4, accum=2, train_steps=60)   # 32K tokens por paso, como el banco v1
R["K"] = run("K-signo", timeout_min=40, **TRAIN, balance="sign", gamma=1e-3)
R["L"] = run("L-smebu-1e-2", timeout_min=40, **TRAIN, balance="smebu", lam=1e-2)
R["M"] = run("M-smebu-1e-3", timeout_min=40, **TRAIN, balance="smebu", lam=1e-3)'''),
    md("## Resumen"),
    code(r'''import pandas as pd
df = pd.DataFrame([json.loads(l) for l in open(RESULTS)])
cols = ["tag", "status", "seq", "micro", "moe_impl", "residual", "reduce", "prefetch", "momentum_device",
        "tokens_per_s", "mfu", "peak_alloc_GiB", "peak_reserved_GiB", "hours_for_25B", "weeks_at_140h",
        "loss_scale_skips", "ckpt_save_s", "ckpt_GiB", "ckpt_restore_exact"]
display(df[[c for c in cols if c in df.columns]])
base = next((R[k] for k in ("A", "A4") if ok(R.get(k))), None)
if base:
    print(f"Base: {base['tokens_per_s']:,} tokens/s, pico reservado {base['peak_reserved_GiB']} GiB, "
          f"~{base['hours_for_25B']:,} h para 25B ({base['weeks_at_140h']} semanas a 140 h/semana)")
    for k, nombre in [("B", "loop en vez de bmm"), ("C", "residuo FP16"), ("D", "gradientes FP32"),
                      ("E", "sin prefetch"), ("F", "momento en CPU"), ("G", "4 secuencias por GPU")]:
        r = R.get(k)
        if k not in R:
            continue
        if ok(r):
            print(f"  {k} {nombre:22s}: {100 * (r['tokens_per_s'] / base['tokens_per_s'] - 1):+5.1f}% tokens/s, "
                  f"{r['peak_reserved_GiB'] - base['peak_reserved_GiB']:+.2f} GiB reservados")
        else:
            print(f"  {k} {nombre:22s}: {(r or {}).get('status')}")
for k in ["H", "I", "J"]:
    r = R.get(k) or {}
    print(f"  {k}: {r.get('tag')} -> {r.get('status')}, {r.get('tokens_per_s')} tokens/s, {r.get('peak_reserved_GiB')} GiB"
          + (f", checkpoint {r.get('ckpt_GiB')} GiB, exacto={r.get('ckpt_restore_exact')}" if k == "J" else ""))
print("\nBalanceo (60 pasos):")
for k in ["K", "L", "M"]:
    r = R.get(k) or {}
    if ok(r):
        print(f"  {r['tag']:14s}: pérdida {r['ce_first']:.2f} -> {r['ce_last']:.2f} (piso ~{r['ce_floor']}), "
              f"carga máx/media al final {r['load_max_over_mean_last']:.2f} (pico {r['load_max_over_mean_peak']:.2f}), "
              f"pasos saltados {r['loss_scale_skips']}, {r['tokens_per_s']:,.0f} tokens/s")
    else:
        print(f"  {k}: {r.get('status')}")'''),
    md("""## Qué hacer con el resultado

Descarga `results/t0_2v2.jsonl` (panel *Output*), guárdalo en `results/` y anota la entrada en la bitácora. Con esos números se fijan las piezas que van al entrenamiento real, el micro-lote, si 16K cabe y la regla de balanceo. Después vienen la T0.3 (estabilidad en FP16 con el LR pico) y la compuerta G0."""),
]

for nb, name in [(t01, "t0_1_formato_mtp_tokenizador.ipynb"), (t02, "t0_2_velocidad_memoria_2xT4.ipynb"),
                 (t02v2, "t0_2v2_afmoe_2xT4.ipynb")]:
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:02d}"          # stable ids: regenerating gives an identical file
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote notebooks/" + name)
