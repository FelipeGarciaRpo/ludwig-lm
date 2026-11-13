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

# ----------------------------------------------------------------------------- T0.3 + T2 (training loop)
TRAIN_HELPER = r'''import threading, time, glob, shutil
KEEP = ("[model]", "[data]", "[train]", "[eval]", "[save]", "[resume]", "[alarm]", "[end]", "[checkpoint]",
        "Traceback", "Error", "out of memory")

def run(tag, out, timeout_h, **kw):
    """Lanza scripts/train.py con torchrun en las 2 GPUs, muestra el progreso y devuelve su fila final."""
    def arg(k, v):
        flag = "--" + k.replace("_", "-")
        return flag if v is True else f"{flag} {v}"
    args = " ".join(arg(k, v) for k, v in kw.items() if v is not False and v != "")
    cmd = (f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 2 "
           f"scripts/train.py --tag {tag} --out {out} {args}")
    print(">>", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    killed = []
    def kill():  # último recurso: train.py se detiene solo con --max-hours
        killed.append(True)
        subprocess.run(["pkill", "-TERM", "-f", "scripts/train.py"])
        time.sleep(30)
        subprocess.run(["pkill", "-KILL", "-f", "scripts/train.py"])
    timer = threading.Timer(timeout_h * 3600, kill)
    timer.start()
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if any(k in line for k in KEEP):
            print(line, end="", flush=True)
    p.wait()
    timer.cancel()
    if p.returncode != 0:
        print(f"[{tag}] terminó con código {p.returncode}" + (" (se pasó del tiempo)" if killed else "")
              + "; últimas líneas:\n" + "".join(tail[-15:]))
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    return next((r for r in reversed(rows) if r.get("tag") == tag and r.get("type") == "end"),
                {"tag": tag, "status": "sin fila final"})'''

t03 = nbf.v4.new_notebook()
t03.cells = [
    md("""# Ludwig 2 · T0.3 + T2: learning rate y estabilidad en FP16, en 2×T4

Entrena el **modelo real** (1.906B totales, 0.491B activos) con `scripts/train.py`, el bucle de entrenamiento de verdad: el lector de datos con reanudación exacta, el schedule WSD, la evaluación en bits por byte, las alarmas del diseño (5.4) y los checkpoints con reanudación verificada. Usa las piezas que eligió la T0.2 v2: un matmul por experto, residuo en FP32, gradientes en FP16 entre las GPUs, prefetch y el momento de Muon en la RAM de la CPU.

**Son tres corridas iguales que solo cambian el learning rate**, una por cuenta y en paralelo:

| Cuenta | `CORRIDA` | `LR_MULT` | Muon / AdamW |
|---|---|---|---|
| cuenta 1 | `lr-x0.5` | 0.5 | 5e-4 / 1.5e-4 |
| cuenta 2 | `lr-x1` | 1.0 | 1e-3 / 3e-4 |
| cuenta 3 | `lr-x2` | 2.0 | 2e-3 / 6e-4 |

Cada una hace **400 pasos de 0.25M tokens (~105M tokens)**: 50 de calentamiento, LR constante y un enfriamiento 1−√ en los últimos 80 pasos, con el mismo orden de datos. Se comparan los bits por byte de validación al final. La de ×2 es además la prueba de estabilidad en FP16 con margen.

**Datos y tokenizador provisionales:** FineWeb-Edu (`sample-10BT`, archivos 000 y 001 para entrenar y 013 para validar) y el tokenizador de SmolLM2 (49,152 tokens, Apache 2.0, con `<|endoftext|>` = 0 como el nuestro), mientras se termina el de 64K. Por eso la métrica son los bits por byte, que no dependen del tokenizador.

**Kaggle:** acelerador *GPU T4 x2*, Internet *On*, el repo agregado como dataset. *Save Version → Save & Run All*. Tarda **~9.5 h** de las 12 que da la sesión; si se acaba el tiempo, guarda un checkpoint y para limpio."""),
    code(r'''CORRIDA = "lr-x1"      # lr-x0.5 · lr-x1 · lr-x2: una por cuenta
LR_MULT = 1.0          # 0.5 · 1.0 · 2.0
REPO_URL = ""          # "" = el repo subido como dataset de Kaggle (el repo es privado)
PASOS, CALENTAMIENTO, ENFRIAMIENTO = 400, 50, 80
GUARDAR_CHECKPOINT = True   # deja en la salida el checkpoint de antes del enfriamiento (~11.3 GB), para seguir la corrida
REANUDAR = ""          # para seguir una corrida: la carpeta del checkpoint, p. ej. "/kaggle/input/<notebook>/checkpoint"
import time
T_SESION = time.time()'''),
    code(SETUP + "\n!pip install -q hf_xet\n!nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv\n"
         "!df -h /tmp /kaggle/working | tail -2\n!free -g | head -2"),
    md("## 1. Datos y tokenizador\n\nSe bajan de Hugging Face sin token (son públicos): ~4.8 GB."),
    code(r'''from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
t = time.time()
DATOS = "/tmp/ludwig_datos"
TRAIN = [hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet", repo_type="dataset",
                         local_dir=DATOS) for i in (0, 1)]
VAL = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/013_00000.parquet", repo_type="dataset", local_dir=DATOS)
TOK = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json", local_dir="/tmp/ludwig_tok")
for f in TRAIN + [VAL]:
    m = pq.ParquetFile(f).metadata
    print(f"{os.path.basename(f)}: {os.path.getsize(f) / 1e9:.2f} GB, {m.num_rows:,} documentos, {m.num_row_groups} row groups")
print(f"listo en {time.time() - t:.0f} s")
COMUN = dict(tokenizer=TOK, train=" ".join(TRAIN), val=VAL)'''),
    code(TRAIN_HELPER),
    md("""## 2. Prueba de guardado y reanudación (~5 min)

Dos pasos cortos, guardar, y reanudar en otro proceso. La reanudación compara el estado cargado bit a bit con el que se guardó (pesos, momento de Muon en la CPU, AdamW, bias de los expertos) y el primer lote con el hash guardado: lo que pide la G0. Si algo falla, la corrida del paso 3 sigue igual: solo su checkpoint quedaría en duda."""),
    code(r'''PRUEBA = f"results/t0_3_t2/{CORRIDA}-prueba.jsonl"
CORTA = dict(**COMUN, warmup=1, tokens_per_step=32768, eval_every=0, eval_windows=8)
run("prueba-a", PRUEBA, 0.5, **CORTA, steps=2, save_dir="/tmp/ludwig_prueba", save_final=True)
run("prueba-b", PRUEBA, 0.5, **CORTA, steps=3, resume="/tmp/ludwig_prueba/step_000002")
filas = [r for r in map(json.loads, open(PRUEBA)) if r["time"] >= T_SESION]   # solo esta sesión
res = next((r for r in filas if r["type"] == "resume"), {})
lote = next((r for r in filas if r["type"] == "resume_check"), {})
ev = {(r["tag"], r["step"]): r["val_bpb"] for r in filas if r["type"] == "eval"}
dif = abs(ev.get(("prueba-a", 2), 0) - ev.get(("prueba-b", 2), 9))
print(f"estado idéntico: {res.get('state_exact')} | primer lote idéntico: {lote.get('batch_exact')} | "
      f"validación antes/después de reanudar: {dif:.2e} bits por byte de diferencia")
print("✅ guardado y reanudación exactos" if res.get("state_exact") and lote.get("batch_exact") and dif < 1e-3
      else "⚠️ la reanudación no salió exacta: revisar antes de confiar en los checkpoints")
!rm -rf /tmp/ludwig_prueba'''),
    md("""## 3. La corrida

Imprime una línea cada 10 pasos y una evaluación cada 50 (más una antes del enfriamiento y otra al final). Cada paso queda en `results/t0_3_t2/<CORRIDA>.jsonl` apenas termina. Se detiene sola antes de las 12 h de la sesión, guardando un checkpoint."""),
    code(r'''SALIDA = f"results/t0_3_t2/{CORRIDA}.jsonl"
horas = round(11.3 - (time.time() - T_SESION) / 3600, 2)
kw = dict(**COMUN, steps=PASOS, warmup=CALENTAMIENTO, decay_steps=ENFRIAMIENTO, lr_mult=LR_MULT,
          eval_every=50, save_dir="/tmp/ludwig_ckpt", max_hours=horas)
if REANUDAR:
    kw["resume"] = REANUDAR
print(f"{CORRIDA}: LR ×{LR_MULT}, {PASOS} pasos, tope {horas} h")
fin = run(CORRIDA, SALIDA, horas + 0.3, **kw)
print(fin)'''),
    md("## 4. Resumen"),
    code(r'''import pandas as pd
import matplotlib.pyplot as plt
filas = [r for r in map(json.loads, open(SALIDA)) if r["tag"] == CORRIDA and r["time"] >= T_SESION]
pasos = pd.DataFrame([r for r in filas if r["type"] == "step"])
evals = pd.DataFrame([r for r in filas if r["type"] == "eval"])
fin = ([r for r in filas if r["type"] == "end"] or [{}])[-1]
if len(evals):
    display(evals[["step", "tokens", "val_bpb", "val_ce", "router_entropy", "qk_gain_max", "dead_experts_window"]])
if fin.get("status") in (None, "error"):
    print("⚠️ la corrida no terminó bien: ver el log de arriba", fin.get("error", ""))
else:
    n = lambda x: "—" if x is None else f"{x:,}"
    print(f"{CORRIDA}: {fin['status']}, paso {fin['steps_done']}, {fin['tokens'] / 1e6:.0f}M tokens en "
          f"{fin['hours_this_session']:.2f} h")
    print(f"  val bpb antes del enfriamiento {fin['val_bpb_pre_decay']} -> al final {fin['val_bpb_final']}")
    print(f"  FP16: {fin['loss_scale_skips']} pasos saltados, escala final {fin['loss_scale_last']:.0f}; "
          f"picos de pérdida saltados: {fin['spikes_skipped']}; alarmas: {fin['alarms']}")
    tps, pico = fin["tokens_per_s_last_hour"], fin["peak_reserved_GiB"]
    print(f"  velocidad sostenida (última hora): {n(tps)} tokens/s, MFU {fin['mfu']}, ~{n(fin['hours_for_25B'])} h "
          f"para 25B | G0 pide ≥ 2,600: {'✅' if tps and tps >= 2600 else '❌'}")
    print(f"  memoria pico: {pico} GiB reservada, {fin['peak_alloc_GiB']} asignada | G0 pide ≤ 13.5 reservada: "
          f"{'✅' if pico <= 13.5 else '❌'}")
for r in filas:
    if r["type"] == "alarm":
        print("  alarma:", r["step"], r["alarm"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(pasos.tokens / 1e6, pasos.ce.rolling(10, min_periods=1).mean())
ax[0].set(xlabel="millones de tokens", ylabel="nats por token", title=f"{CORRIDA}: pérdida de entrenamiento (media de 10 pasos)")
ax[1].plot(evals.tokens / 1e6, evals.val_bpb, "o-")
ax[1].set(xlabel="millones de tokens", ylabel="bits por byte", title="validación",
          ylim=(None, evals.val_bpb.iloc[1:].max() * 1.05) if len(evals) > 1 else (None, None))
ax[2].plot(pasos.step, pasos.load_max_over_mean.rolling(10, min_periods=1).mean())
ax[2].axhline(3, ls="--", c="gray")
ax[2].set(xlabel="paso", ylabel="experto más cargado / promedio", title="balanceo (media de 10 pasos; alarma en 3)")
fig.tight_layout()
fig.savefig(f"results/t0_3_t2/{CORRIDA}.png", dpi=120)
plt.show()'''),
    code(r'''# El checkpoint de antes del enfriamiento (paso 320) a la salida del notebook, para poder seguir esta corrida
if GUARDAR_CHECKPOINT:
    hechos = sorted(os.path.dirname(m) for m in glob.glob("/tmp/ludwig_ckpt/step_*/meta.json"))
    pre = [d for d in hechos if json.load(open(f"{d}/meta.json"))["reason"] == "pre_decay"]
    elegido = (pre or hechos or [None])[-1]
    if elegido:
        shutil.rmtree("/kaggle/working/checkpoint", ignore_errors=True)
        shutil.copytree(elegido, "/kaggle/working/checkpoint")
        print("checkpoint en la salida:", elegido, "->", "/kaggle/working/checkpoint")
    else:
        print("no hay checkpoint que guardar")
!du -sh /kaggle/working/checkpoint 2>/dev/null; ls results/t0_3_t2/'''),
    md("""## Qué hacer con el resultado

1. Descarga de la salida del notebook `results/t0_3_t2/` (el `.jsonl` de la corrida, el de la prueba y el `.png`) y el log de Kaggle, guárdalos en `results/t0_3_t2/` y anota la entrada en la bitácora.
2. **Regla de decisión, fijada antes de ver los resultados:**
   - Una corrida cuenta solo si fue estable: sin NaN, sin parar por picos y con la escala de FP16 estable.
   - Gana la de menor bits por byte de validación **al final** (después del enfriamiento). Si dos quedan a menos de 0.5%, gana la de LR más bajo: las corridas cortas favorecen los LR altos, y el tronco es 200 veces más largo.
   - Si gana un extremo (×0.5 o ×2), se corre uno más allá (×0.25 o ×4) antes de decidir.
   - SMEBU con λ = 0.01 queda si la carga máx/media se mantiene bajo 3 y no hay expertos muertos.
3. **Para seguir una corrida** (por ejemplo, alargar la ganadora como prueba de estabilidad de la T0.3): agrega la salida de esta versión como dataset de entrada, pon en `REANUDAR` la carpeta `checkpoint`, sube `PASOS` (p. ej. 2000) y pon `ENFRIAMIENTO = 0`. La corrida sigue desde el paso 320 con el mismo orden de datos."""),
]

for nb, name in [(t01, "t0_1_formato_mtp_tokenizador.ipynb"), (t02, "t0_2_velocidad_memoria_2xT4.ipynb"),
                 (t02v2, "t0_2v2_afmoe_2xT4.ipynb"), (t03, "t0_3_t2_lr_2xT4.ipynb")]:
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:02d}"          # stable ids: regenerating gives an identical file
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote notebooks/" + name)
