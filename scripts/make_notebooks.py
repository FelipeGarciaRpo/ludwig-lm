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
TRAIN_HELPER = r'''import threading, time, glob, shutil, re
DUENO = re.compile(r"(/kaggle/input/(?:notebooks|datasets|models)/)[^/\"\s]+")
publico = lambda texto: DUENO.sub(r"\1<usuario>", texto)   # el usuario de Kaggle de una entrada no va a los logs
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
    print(">>", publico(cmd), flush=True)
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
            print(publico(line), end="", flush=True)
    p.wait()
    timer.cancel()
    if p.returncode != 0:
        print(f"[{tag}] terminó con código {p.returncode}" + (" (se pasó del tiempo)" if killed else "")
              + "; últimas líneas:\n" + publico("".join(tail[-15:])))
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
| cuenta 1 | `lr-x0.5` | 0.5 | 7.5e-4 / 2.5e-4 |
| cuenta 2 | `lr-x1` | 1.0 | 1.5e-3 / 5e-4 |
| cuenta 3 | `lr-x2` | 2.0 | 3e-3 / 1e-3 |

**×1 es la receta del Ludwig 1:** AdamW 5e-4 para embeddings, normas y router; Muon 3 veces eso, 1.5e-3 (en el A/B del 31 de octubre el ×3 le ganó a AdamW por 0.37 nats); weight decay 0.1 aplicado con el LR de AdamW también a las matrices de Muon. Los dos Muon usan la misma escala (la de Moonlight), así que los números se pueden comparar directamente.

Cada una hace **400 pasos de 0.25M tokens (~105M tokens)**: 100 de calentamiento, LR constante y un enfriamiento 1−√ en los últimos 80 pasos (el último 20%, como en el Ludwig 1), con el mismo orden de datos. Se comparan los bits por byte de validación al final. La de ×2 es además la prueba de estabilidad en FP16 con margen.

**Datos y tokenizador provisionales:** FineWeb-Edu (`sample-10BT`, archivos 000 y 001 para entrenar y 013 para validar) y el tokenizador de SmolLM2 (49,152 tokens, Apache 2.0, con `<|endoftext|>` = 0 como el nuestro), mientras se termina el de 64K. Por eso la métrica son los bits por byte, que no dependen del tokenizador.

**Kaggle:** acelerador *GPU T4 x2*, Internet *On*, el repo agregado como dataset. *Save Version → Save & Run All*. Tarda **~9.5 h** de las 12 que da la sesión; si se acaba el tiempo, guarda un checkpoint y para limpio.

**Segunda ronda (regla P9 de `docs/DECISIONS.md`).** En la primera ganó ×2. Las corridas cortas favorecen los LR altos, así que la segunda mide cuánto se mueve el mejor LR cuando la corrida es más larga:

| Cuenta | `CORRIDA` | `LR_MULT` | `LAMBDA` | `PASOS, CALENTAMIENTO, ENFRIAMIENTO` | `REANUDAR` | `PRUEBA` | Sesiones |
|---|---|---|---|---|---|---|---|
| una libre | `lr-x4` | 4.0 | 0.04 | 400, 100, 80 | `""` | `True` | 1 |
| la que corrió `lr-x1` | `lr-x1-1200` | 1.0 | 0.01 | 1200, 100, 240 | `"auto"` | `False` | 2 |
| la que corrió `lr-x2` | `lr-x2-1200` | 2.0 | 0.01 | 1200, 100, 240 | `"auto"` | `False` | 2 |

- `lr-x4` cierra el barrido corto, con el balanceo de SMEBU escalado con el LR (λ = 0.01 × el multiplicador).
- Las otras dos **siguen** las corridas ×1 y ×2 desde su checkpoint del paso 320 hasta 1,200 pasos, 3 veces más largas, con el mismo λ que tuvieron.
- Para seguir una corrida, agrega como entrada (*Add Input → Your Work*) la salida de la versión que la corrió. Con `REANUDAR = "auto"` el notebook busca ahí el checkpoint de esa corrida más avanzado.
- La primera sesión para sola hacia el paso 800 y deja su checkpoint en la salida. La segunda se lanza igual, con la salida de la primera como entrada, y termina en el 1,200."""),
    code(r'''CORRIDA = "lr-x1"      # lr-x0.5 · lr-x1 · lr-x2: una por cuenta
LR_MULT = 1.0          # 0.5 · 1.0 · 2.0
REPO_URL = ""          # "" = el repo subido como dataset de Kaggle (el repo es privado)
PASOS, CALENTAMIENTO, ENFRIAMIENTO = 400, 100, 80
LAMBDA = 0.01          # SMEBU: 0.01 × LR_MULT en una corrida nueva; el mismo de antes en una que sigue
GUARDAR_CHECKPOINT = True   # deja en la salida el checkpoint de antes del enfriamiento (~11.3 GB), para seguir la corrida
REANUDAR = ""          # para seguir una corrida: "auto" (lo busca en los datos de entrada) o la carpeta del checkpoint
PRUEBA = True          # la prueba de guardado y reanudación (~5 min); al seguir una corrida sobra: se verifica al reanudar
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
    code(r'''if PRUEBA:
    ARCHIVO_PRUEBA = f"results/t0_3_t2/{CORRIDA}-prueba.jsonl"
    CORTA = dict(**COMUN, warmup=1, tokens_per_step=32768, eval_every=0, eval_windows=8)
    run("prueba-a", ARCHIVO_PRUEBA, 0.5, **CORTA, steps=2, save_dir="/tmp/ludwig_prueba", save_final=True)
    run("prueba-b", ARCHIVO_PRUEBA, 0.5, **CORTA, steps=3, resume="/tmp/ludwig_prueba/step_000002")
    filas = [r for r in map(json.loads, open(ARCHIVO_PRUEBA)) if r["time"] >= T_SESION]   # solo esta sesión
    res = next((r for r in filas if r["type"] == "resume"), {})
    lote = next((r for r in filas if r["type"] == "resume_check"), {})
    ev = {(r["tag"], r["step"]): r["val_bpb"] for r in filas if r["type"] == "eval"}
    dif = abs(ev.get(("prueba-a", 2), 0) - ev.get(("prueba-b", 2), 9))
    print(f"estado idéntico: {res.get('state_exact')} | primer lote idéntico: {lote.get('batch_exact')} | "
          f"validación antes/después de reanudar: {dif:.2e} bits por byte de diferencia")
    print("✅ guardado y reanudación exactos" if res.get("state_exact") and lote.get("batch_exact") and dif < 1e-3
          else "⚠️ la reanudación no salió exacta: revisar antes de confiar en los checkpoints")
    shutil.rmtree("/tmp/ludwig_prueba", ignore_errors=True)
else:
    print("sin prueba: esta corrida sigue un checkpoint, y su reanudación se verifica al cargarlo")'''),
    md("""## 3. La corrida

Imprime una línea cada 10 pasos y una evaluación cada 50 (más una antes del enfriamiento y otra al final). Cada paso queda en `results/t0_3_t2/<CORRIDA>.jsonl` apenas termina. Se detiene sola antes de las 12 h de la sesión, guardando un checkpoint."""),
    code(r'''SALIDA = f"results/t0_3_t2/{CORRIDA}.jsonl"
horas = round(11.3 - (time.time() - T_SESION) / 3600, 2)
kw = dict(**COMUN, steps=PASOS, warmup=CALENTAMIENTO, decay_steps=ENFRIAMIENTO, lr_mult=LR_MULT, lam=LAMBDA,
          eval_every=50, save_dir="/tmp/ludwig_ckpt", max_hours=horas)
if REANUDAR == "auto":
    # los checkpoints de esta corrida, o de la que sigue ("lr-x1" para "lr-x1-1200"), en los datos de entrada
    hay = []
    for m in glob.glob("/kaggle/input/**/checkpoint/meta.json", recursive=True):
        meta = json.load(open(m))
        if meta["tag"] == CORRIDA or CORRIDA.startswith(meta["tag"] + "-"):
            hay.append((meta["step"], os.path.dirname(m), meta["tag"], meta["reason"]))
    if not hay:
        raise SystemExit(f"no hay checkpoint de {CORRIDA} en los datos de entrada: agrega la salida de la versión que la corrió")
    paso, REANUDAR, de, motivo = max(hay)
    print(f"sigue {de} desde el paso {paso} ({motivo}): {publico(REANUDAR)}")
    # si es la segunda sesión de esta misma corrida, su .jsonl sigue en el mismo archivo
    antes = glob.glob(os.path.join(os.path.dirname(REANUDAR), "**", "results", "t0_3_t2", f"{CORRIDA}.jsonl"),
                      recursive=True)
    if antes and not os.path.exists(SALIDA):
        shutil.copy(antes[0], SALIDA)
        print(f"{SALIDA}: sigue el de la sesión anterior ({sum(1 for _ in open(SALIDA))} filas)")
if REANUDAR:
    kw["resume"] = REANUDAR
print(f"{CORRIDA}: LR ×{LR_MULT}, λ {LAMBDA}, {PASOS} pasos, tope {horas} h")
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
3. **Segunda ronda (P9):** con los `.jsonl` de `lr-x4`, `lr-x1-1200` y `lr-x2-1200` en `results/t0_3_t2/`, `python scripts/lr_horizon.py` da el LR del tronco. Mide cuánto baja el mejor LR de 400 a 1,200 pasos y lo lleva a los 25B tokens del tronco.
4. **Para seguir una corrida:** agrega como entrada la salida de la versión que la corrió y pon `REANUDAR = "auto"`. Sube `PASOS` y pon el `ENFRIAMIENTO` que toque. La corrida sigue con el mismo orden de datos, desde el checkpoint más avanzado de esa corrida que encuentre."""),
]

# ----------------------------------------------------------------------------- T1 (shards, CPU)
T1_RUN = r'''import subprocess, time, json, glob, os, threading
LATIDO = 300             # cada cuántos segundos imprime una línea de avance (0 = nunca)

def _red():
    """Bytes recibidos por la red desde que arrancó la máquina, sin contar la interfaz local."""
    total = 0
    for linea in open("/proc/net/dev"):
        if ":" in linea:
            nombre, datos = linea.split(":", 1)
            if nombre.strip() != "lo":
                total += int(datos.split()[0])
    return total

def _cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]                       # todo, y el tiempo ocioso o esperando disco

def _latido(salida, fin, t0):
    """Mientras corre el constructor: la red, la CPU, la RAM y lo que ya dejó en la salida. Algunas etapas
    (las listas de Stack-Edu en prepare) trabajan mucho rato sin imprimir nada."""
    r0, (c0, i0), antes = _red(), _cpu(), time.time()
    while not fin.wait(LATIDO):
        r1, (c1, i1), ahora = _red(), _cpu(), time.time()
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        ram = (mem["MemTotal"] - mem["MemAvailable"]) / 2**20
        cpu = 100 * (1 - (i1 - i0) / max(c1 - c0, 1))
        prep = sorted(os.path.basename(p) for p in glob.glob(f"{salida}/prep/*"))
        shards = len(glob.glob(f"{salida}/train/shard_*.bin"))
        usado = sum(os.path.getsize(p) for p in glob.glob(f"{salida}/**/*", recursive=True) if os.path.isfile(p)) / 2**30
        h, m = divmod(int((ahora - t0) / 60), 60)
        print(f"⏱ {h}h{m:02d} · bajando {(r1 - r0) / (ahora - antes) / 1e6:.1f} MB/s ({r1 / 1e9:.1f} GB en total) · "
              f"CPU {cpu:.0f}% · RAM {ram:.1f} GB · salida {usado:.1f} GB · prep: {', '.join(prep) or '—'} · "
              f"shards: {shards}", flush=True)
        r0, c0, i0, antes = r1, c1, i1, ahora

def correr(*args):
    """Corre el constructor y va mostrando su salida; si falla, para el notebook. Cada LATIDO segundos suma
    una línea ⏱ con el avance."""
    print(">>", " ".join(args), flush=True)
    salida = args[args.index("--out") + 1] if "--out" in args else "/kaggle/working/t1"
    salida = os.path.dirname(salida) if salida.endswith(".txt") else salida
    fin = threading.Event()
    if LATIDO:
        threading.Thread(target=_latido, args=(salida, fin, time.time()), daemon=True).start()
    p = subprocess.Popen(["python", "scripts/build_shards.py", *args], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        for linea in p.stdout:
            print(linea, end="", flush=True)
    finally:
        fin.set()
    if p.wait() != 0:
        raise SystemExit(f"build_shards.py terminó con código {p.returncode}")'''

t1 = nbf.v4.new_notebook()
t1.cells = [
    md("""# T1 · Shards pre-tokenizados del tronco (Kaggle, CPU)

Convierte las fuentes de la mezcla v3 (`configs/data_v3.json`, DISENO 4.4) en **shards de 256 MiB de ids uint16** que el entrenamiento solo lee: la sesión de GPU no descarga, no filtra ni tokeniza.

Qué hace con cada documento: el formato de su fuente, descontaminación contra 16 benchmarks (13-gramas exactos), FIM en el 10% del código, la plantilla de chat con `<think>` como id en las trazas, la etiqueta MeCo de la fuente, y la mezcla exacta por tokens. Una página nunca entra dos veces entre fuentes ni entre sus formas (original, reescritura, traducción).

**Kaggle:** sin acelerador (CPU), Internet *On*, y en *Add-ons → Secrets* un `HF_TOKEN` de la cuenta que aceptó los términos de `bigcode/the-stack-dedup`, `bigcode/starcoderdata` y `openlanguagedata/flores_plus`. *Save Version → Save & Run All*.

**Las sesiones** (el tronco de la v1 son 168 shards = 22.5B tokens). La primera del build `75cd3a0` hizo 28 shards en 10.4 h (22 min por shard) y paró por el tope de horas, no por `SHARDS`; a ese ritmo son unas 6 sesiones de `construir`:

| Sesión | `FASE` | Datos de entrada | Tarda | Sale |
|---|---|---|---|---|
| 0 | `plan` | ninguno | ~10 min | `plan.json`: cada fuente y cada benchmark contra el Hub |
| 1 | `construir` | ninguno | ~11 h | descontaminación, listas de Stack-Edu, validación y ~28 shards |
| 2, 3, … | `construir` | la salida de la sesión anterior (como dataset) | ~11 h | ~28 shards más cada una |

El tronco gasta ~1 shard por turno, así que la T1 va muy por delante: sus sesiones (CPU) pueden correr mientras corre el tronco (GPU).

- **El commit queda fijo** en `COMMIT` para todas las sesiones. El constructor se niega a seguir con otro código, porque cambiaría los shards.
- Si `plan` marca ❌, se corrige en `configs/data_v3.json`, no en el código. Por ejemplo, un nombre de columna o de carpeta. Después se vuelve a correr `plan`.
- Cada sesión deja en la salida lo que necesita la siguiente: el estado, la validación y la descontaminación. Si una sesión muere, se pierde a lo más un shard."""),
    code(r'''FASE = "plan"            # plan · construir · muestras
COMMIT = "75cd3a08b5e3195908c7aff769b4a0254f2b20e6"  # el MISMO en todas las sesiones de la T1 (bitácora: el commit vigente)
REPO_URL = "https://github.com/FelipeGarciaRpo/ludwig-lm"
SHARDS = 56              # por sesión: 56 × 256 MiB caben en los 20 GB de salida de Kaggle
HORAS = 11.0             # tope de la sesión, contando desde aquí
OTRO_COMMIT = False      # True solo si la bitácora lo indica: sigue un build empezado con otro commit, cuando el
                         # cambio no puede alterar los shards (--allow-code-change)
import time
T_SESION = time.time()'''),
    code(r'''import os, subprocess, sys
os.chdir("/kaggle/working")
if not os.path.exists("ludwig-lm"):
    subprocess.run(["git", "clone", "-q", REPO_URL, "ludwig-lm"], check=True)
subprocess.run(["git", "-C", "ludwig-lm", "checkout", "-q", COMMIT], check=True)
os.chdir("/kaggle/working/ludwig-lm")
print("commit", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
!pip install -q orjson hf_xet
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN: sí")
except Exception as e:
    print("⚠️ sin HF_TOKEN: The Stack v1, StarCoderData y Flores+ no van a abrir", type(e).__name__)
import pyarrow, fsspec, huggingface_hub
print("pyarrow", pyarrow.__version__, "· fsspec", fsspec.__version__, "· huggingface_hub", huggingface_hub.__version__)
!nproc; free -g | head -2; df -h /kaggle/working | tail -1'''),
    code(T1_RUN),
    md("""## 1. Plan

Lee el primer archivo de cada fuente y de cada benchmark y arma su primer documento. Así muestra qué archivos encontró, qué columnas trae cada uno y un ejemplo de texto. No construye nada."""),
    code(r'''SALIDA = "/kaggle/working/t1"
if FASE == "plan":
    try:
        correr("plan", "--out", SALIDA)
    finally:
        plan = json.load(open(f"{SALIDA}/plan.json"))
        for n, r in plan["sources"].items():
            if r["status"] == "ok":
                print(f"\n✅ {n}: {r['files']} archivos, {r['bytes'] / 1e9:.1f} GB · columnas {list(r['columns'])[:12]}")
                print(f"   {r['example'][:200]!r}")
            else:
                print(f"\n❌ {n}: {r['error']}")
else:
    print("FASE no es plan")'''),
    md("""## 2. Construcción

La primera sesión prepara la descontaminación y las listas de Stack-Edu, escribe la validación, y después construye shards. Las siguientes buscan solas el estado más avanzado en los datos de entrada y siguen desde ahí, exacto."""),
    code(r'''if FASE == "construir":
    horas = round(HORAS - (time.time() - T_SESION) / 3600 - 0.3, 2)
    correr("build", "--out", SALIDA, "--from", "auto", "--roots", "/kaggle/input", "--shards", str(SHARDS),
           "--max-hours", str(horas), *(["--allow-code-change"] if OTRO_COMMIT else []))
    correr("samples", "--manifest", f"{SALIDA}/manifest.json", "--out", f"{SALIDA}/muestras.txt",
           "--roots", "/kaggle/input")
    rep = json.load(open(sorted(glob.glob(f"{SALIDA}/report_*.json"))[-1]))
    receta = {s["name"]: s["weight"] for s in json.load(open(f"{SALIDA}/recipe.json"))["sources"]}
    print(f"\n{rep['session_shards']} shards en esta sesión, {rep['total_shards']} en total, {rep['hours']} h "
          f"({rep['minutes_per_shard']} min por shard)")
    print(f"{'fuente':22s} {'% real':>7s} {'receta':>7s} {'docs':>9s} {'repetidos':>9s} {'contam.':>8s} {'val':>6s} {'largos':>7s} {'FIM':>6s}")
    for n, pct in rep["share_by_source"].items():
        st = rep["stats"][n]
        print(f"{n:22s} {pct:7.2f} {receta[n]:7.2f} {st.get('docs', 0):9,} {st.get('dup', 0):9,} {st.get('decon', 0):8,} "
              f"{st.get('val_excluded', 0):6,} {st.get('too_long', 0):7,} {st.get('fim', 0):6,}")
elif FASE == "muestras":
    correr("samples", "--manifest", sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))[-1],
           "--out", "/kaggle/working/muestras.txt", "--roots", "/kaggle/input")
!du -sh /kaggle/working/t1 2>/dev/null; ls /kaggle/working/t1 2>/dev/null'''),
    md("""## Qué hacer con el resultado

1. **Después de cada sesión de `construir`:**
   - En la salida del notebook: *New Dataset*, privado, con nombre `ludwig2-shards-1` (después `-2`, `-3`, …). Nunca los nombres del build viejo (`ludwig2-tronco-v3-*`), que se reemplazó.
   - Comparte el dataset con las cuentas del relevo (*Share → can view*).
   - La siguiente sesión lo agrega como entrada (*Add Input*).
2. **Al repo,** en `results/t1/` y con su fila en `results/README.md`:
   - `plan.json`, `manifest.json`, los `report_*.json` y `muestras.txt` de cada sesión;
   - `prep/prepare.json`, que cuenta los 13-gramas de cada benchmark y los archivos permisivos de Stack-Edu.

   Los shards no van al repo, porque tienen texto y código de terceros; quedan privados en Kaggle. El manifiesto, con los sha256, sí va.
3. **Antes de la G0:**
   - Lee `muestras.txt`: tres documentos de cada fuente, decodificados con sus tokens de control. Ahí se ve si algo salió mal: un formato raro, etiquetas mal puestas, FIM o plantilla de chat rotos.
   - Revisa en el último reporte que el % real de cada fuente esté a menos de 0.1 punto de la receta, y que las columnas de repetidos y contaminación tengan sentido."""),
]

# ----------------------------------------------------------------------------- the trunk's relay (2×T4)
tronco = nbf.v4.new_notebook()
tronco.cells = [
    md("""# Tronco · un turno del relevo (Kaggle 2×T4)

Un turno: toma el candado en Hugging Face, sigue desde el checkpoint más nuevo (de Hugging Face o de uno de los dos respaldos de Kaggle), entrena ~11 h, sube un checkpoint cada 3 h mientras entrena, deja el último en el respaldo que toca, aplasta el historial y suelta el candado (`ludwig_lm/relay.py`, D15, P26).

**Una vez por cuenta:**
1. Cuenta de Hugging Face, invitada a la organización del proyecto con rol *write*. Un token *fine-grained* con escritura solo en el repo del relevo.
2. En Kaggle, *Add-ons → Secrets*: `HF_TOKEN` (ese token), y `KAGGLE_USERNAME` y `KAGGLE_KEY` (el token de la API de tu cuenta, *Settings → API*). Cada uno usa el suyo.
3. Pipe te agrega como colaborador (*can edit*) en los dos datasets de respaldo y como lector en los tres de shards.

**Cada turno en un notebook nuevo,** importado del repo, con sus entradas recién agregadas. Cada respaldo crea una versión nueva de su dataset y borra las viejas (`kaggle datasets version -d`, para no llenar el espacio de Kaggle con checkpoints de 11 GB). Un notebook que ya tenía adjunta una versión vieja la ve como *no longer available*, y sin la última versión de los dos respaldos el turno no sabe cuál toca.

**En cada turno:** acelerador *GPU T4 x2*, Internet *On*, y como entradas todos los datasets de shards (`ludwig2-shards-1`, `-2`, …) y los dos de respaldo. Nunca los del build viejo (`ludwig2-tronco-v3-*`): el notebook elige el manifiesto con más shards, y podría quedarse con el viejo. Pon tu `CUENTA` ("cuenta 4", nunca el usuario) y *Save Version → Save & Run All*. Avisa por WhatsApp al empezar y al terminar.

| `MODO` | Qué hace | Tarda |
|---|---|---|
| `tronco` | Un turno del tronco | ~11.5 h |
| `prueba-relevo` | **G0-0:** el relevo completo con un modelo diminuto, contra `HF_REPO_PRUEBA`: candado, subida, aplastado, respaldo y reanudación exacta. Solo antes del primer turno del tronco | ~15 min |
| `prueba-respaldo` | Sigue desde el respaldo que dejó la última `prueba-relevo`, sin Hugging Face (como `SIN_CANDADO`): lo lee donde Kaggle lo montó y respalda en el otro dataset. Solo antes del primer turno del tronco | ~5 min |
| `g0-16k` | **G0-1:** 50 pasos del modelo real a 16K de contexto, sin guardar (condición 5 de la G0) | ~30–60 min |"""),
    code(r'''MODO = "tronco"          # tronco · prueba-relevo · prueba-respaldo · g0-16k
CUENTA = "cuenta 1"      # tu etiqueta en el relevo: nunca el usuario de Kaggle ni el correo
COMMIT = "3b9c114587b278cf6b4b16b679cb1c925f22c8f1"  # el commit del tronco: el MISMO en todos los turnos
REPO_URL = "https://github.com/FelipeGarciaRpo/ludwig-lm"
HF_REPO = "ludwig-v2-lm/ludwig-v2-lm-1"                       # privado, de la organización
HF_REPO_PRUEBA = "ludwig-v2-lm/ludwig-v2-lm-1-relevo-prueba"  # privado: solo para prueba-relevo
RESPALDO = ["DUENO/ludwig2-respaldo-a", "DUENO/ludwig2-respaldo-b"]   # los dos datasets privados de Pipe
SIN_CANDADO = False      # True solo si Hugging Face está caído y se coordinó por WhatsApp: sigue desde Kaggle
import time
T_SESION = time.time()'''),
    code(r'''import os, subprocess, sys, json, glob, shutil
os.chdir("/kaggle/working")
if not os.path.exists("ludwig-lm"):
    subprocess.run(["git", "clone", "-q", REPO_URL, "ludwig-lm"], check=True)
subprocess.run(["git", "-C", "ludwig-lm", "checkout", "-q", COMMIT], check=True)
os.chdir("/kaggle/working/ludwig-lm")
print("commit", subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
!pip install -q "transformers==5.22.2" safetensors sentencepiece protobuf hf_xet kaggle
from kaggle_secrets import UserSecretsClient
for k in ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
    try:
        os.environ[k] = UserSecretsClient().get_secret(k)
    except Exception:
        print(f"⚠️ falta el secreto {k}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, ".")
from ludwig_lm.relay import HFStore, KaggleBackup, Lock, LocalStore, Session, train_command, Busy
TRONCO = json.load(open("configs/tronco_v1.json"))
manifiestos = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True),
                     key=lambda m: len(json.load(open(m))["shards"]))
MANIFIESTO = manifiestos[-1] if manifiestos else None
print("shards:", f"{len(json.load(open(MANIFIESTO))['shards'])} en {MANIFIESTO.split('/input/')[-1]}" if MANIFIESTO
      else "⚠️ ninguno: agrega los datasets de shards")
libre = shutil.disk_usage("/tmp").free / 2**30
print(f"/tmp: {libre:.0f} GiB libres" + ("" if libre >= 30 else
      " ⚠️ un turno necesita ~23 GiB (dos checkpoints de 11.3 GiB a la vez): avisa antes de entrenar"))
for slug in RESPALDO:                    # lo que Kaggle montó de cada respaldo, también los ocultos (.metadata)
    try:
        nombre, vistos, hay = slug.split("/")[-1], [], False
        for d, _, fs in os.walk("/kaggle/input"):
            partes = d.split("/")
            if nombre in partes:
                hay, dentro = True, "/".join(partes[partes.index(nombre) + 1:])
                vistos += [(os.path.join(dentro, f), os.path.getsize(os.path.join(d, f))) for f in fs]
        print(f"respaldo {nombre}:", "no está adjunto" if not hay else ", ".join(
            f"{p} ({s / 2**20:.0f} MiB)" for p, s in sorted(vistos)[:8]) or "vacío")
    except Exception as e:  # solo informa: nunca detiene el turno
        print(f"respaldo {slug.split('/')[-1]}: no se pudo listar ({type(e).__name__})")
import torch
if not torch.cuda.is_available():         # en CPU, train.py corre ~100 veces más lento y el turno se pierde
    raise SystemExit("⛔ sin GPU: en Settings → Accelerator elige GPU T4 x2 y vuelve a correr")
if MODO in ("tronco", "prueba-relevo", "prueba-respaldo") and any(s.split("/")[0] == "DUENO" for s in RESPALDO):
    raise SystemExit("⛔ RESPALDO todavía dice DUENO: pon el usuario de Kaggle de Pipe en los dos datasets")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'''),
    md("""## G0-0 · `prueba-relevo`

El relevo completo con el modelo diminuto (vocabulario de 64K, unos segundos por paso), contra `HF_REPO_PRUEBA` y los respaldos reales. Cada corrida usa su propia etiqueta (`prueba-relevo-<fecha y hora>`): el tronco la ignora, y la prueba se puede repetir sin limpiar lo que dejó la anterior.

**Solo antes del primer turno del tronco.** Cada versión de un dataset de respaldo guarda un solo checkpoint, así que una vez que empiece el tronco, esta prueba reemplazaría su respaldo.

1. La sesión A empieza de cero y entrena 30 pasos: guarda en el 10 (lo sube mientras entrena) y se detiene en el 20 como si se acabara el tiempo. Sube, respalda, aplasta y suelta.
2. Se prueba el candado: una sesión lo toma, otra recibe *ocupado*, y lo suelta.
3. La sesión B sigue desde Hugging Face hasta el paso 30. Su reanudación debe ser exacta en el estado y en el primer lote.

Al final imprime cada condición con ✅ o ❌. Lo que Kaggle montó de cada respaldo sale en la celda de preparación."""),
    code(r'''if MODO == "prueba-relevo":
    ETIQUETA = time.strftime("prueba-relevo-%Y%m%d-%H%M%S", time.gmtime())   # nueva en cada corrida
    print("etiqueta:", ETIQUETA)
    c = json.load(open("configs/tiny_afmoe.json")); c.update(vocab_size=65536, pad_token_id=1, bos_token_id=0)
    json.dump(c, open("/tmp/tiny_64k.json", "w"))
    LOG = "/kaggle/working/results/relevo-prueba.jsonl"; os.makedirs(os.path.dirname(LOG), exist_ok=True)
    base = dict(config="/tmp/tiny_64k.json", tokenizer="tokenizer/tokenizer.json", shards=MANIFIESTO,
                shard_roots="/kaggle/input", seq=256, micro=4, tokens_per_step=2048, steps=30, warmup=5,
                eval_every=10, eval_windows=2, log_every=5, save_final=True, tag=ETIQUETA, out=LOG)
    tienda = HFStore(HF_REPO_PRUEBA, os.environ.get("HF_TOKEN"))
    respaldo = KaggleBackup(RESPALDO, tag=ETIQUETA)
    def sesion(extra):
        a = dict(base, **extra)
        # poll_s 0.5: el modelo diminuto va del paso 10 al 20 en ~3 s, y cada 10 s se perdía la subida del 10
        return Session(tienda, respaldo, CUENTA, ETIQUETA, f"/tmp/relevo-{len(extra)}",
                       lambda r, d: train_command(a, r, d), LOG, lease_s=1800, renew_s=600, poll_s=0.5).run()
    A = sesion({"save_at": [10], "pause_at_step": 20})
    l1, l2 = Lock(tienda, "prueba 1", "s1"), Lock(tienda, "prueba 2", "s2")
    h = l1.acquire()
    try:
        l2.acquire(); ocupado = False
    except Busy:
        ocupado = True
    l1.release(h)
    B = sesion({})
    filas = [json.loads(x) for x in open(LOG)]
    res = [r for r in filas if r["type"] == "resume"][-1:]
    lote = [r for r in filas if r["type"] == "resume_check"][-1:]
    subidas = {e["step"] for e in A["events"] + B["events"] if e["kind"] == "upload"}
    for e in A["events"] + B["events"]:
        print(" ", e["kind"], {k: v for k, v in e.items() if k not in ("kind", "time", "by", "session")})
    pruebas = {
        "A empezó de cero": A["resumed_from"]["where"] is None,
        "subió los pasos 10 (mientras entrenaba), 20 y 30": {10, 20, 30} <= subidas,
        "A dejó su checkpoint en un respaldo de Kaggle": any(e["kind"] == "backup" for e in A["events"]),
        "con el candado tomado, otra sesión recibe «ocupado»": ocupado,
        "B siguió desde Hugging Face en el paso 20": B["resumed_from"] == {"where": "hf", "step": 20},
        "reanudación exacta: el estado": bool(res and res[0]["state_exact"]),
        "reanudación exacta: el primer lote": bool(lote and lote[0]["batch_exact"]),
        "A y B aplastaron el historial y soltaron el candado":
            all(k in [e["kind"] for e in S["events"]] for S in (A, B) for k in ("squash", "release")),
    }
    print()
    for k, v in pruebas.items():
        print(" ", "✅" if v else "❌", k)
    ok = all(pruebas.values())
    print("\nG0-0:", "✅ el relevo funciona: subida, candado, respaldo, aplastado y reanudación exacta" if ok
          else "❌ revisar las condiciones y los eventos de arriba")'''),
    md("""## Prueba del respaldo · `prueba-respaldo`

Un turno que sigue desde Kaggle, como con `SIN_CANDADO = True`: sin Hugging Face y sin candado. Toma el respaldo más nuevo de una `prueba-relevo`, lo lee donde Kaggle lo desempacó (sin copiarlo), entrena 10 pasos con el modelo diminuto y deja su checkpoint en el otro dataset de respaldo.

**Solo antes del primer turno del tronco,** por lo mismo que la G0-0: escribe en los respaldos reales, y cada versión guarda un solo checkpoint.

**En un notebook nuevo,** no en el de la `prueba-relevo`: esa prueba crea versiones nuevas de los respaldos y borra las que el notebook tenía adjuntas.

**Entradas:** los dos respaldos y los shards con los que se entrenó ese respaldo. La celda imprime cuáles son: el lector se niega a seguir con shards de otro build.

Pasa si sigue desde Kaggle en el paso del respaldo, sin copiarlo, con el estado y el primer lote exactos, y si respalda en el otro dataset."""),
    code(r'''if MODO == "prueba-respaldo":
    vistos = []
    for slug in RESPALDO:
        nombre = slug.split("/")[-1]
        for m in glob.glob(f"/kaggle/input/**/{nombre}/checkpoint.json", recursive=True):
            meta = json.load(open(m))
            if str(meta.get("tag", "")).startswith("prueba-relevo"):
                vistos.append((meta["step"], meta.get("saved_at", 0), slug, meta))
    if not vistos:
        raise SystemExit("⛔ ningún respaldo de una prueba-relevo montado: adjunta ludwig2-respaldo-a y ludwig2-respaldo-b")
    _, _, slug, meta = max(vistos, key=lambda v: v[:2])
    print(f"respaldo: {slug.split('/')[-1]}, paso {meta['step']}, etiqueta {meta['tag']}")
    print(f"  se entrenó con {meta['args']['shards']}")
    print(f"  shards adjuntos: {MANIFIESTO}")
    c = json.load(open("configs/tiny_afmoe.json")); c.update(vocab_size=65536, pad_token_id=1, bos_token_id=0)
    json.dump(c, open("/tmp/tiny_64k.json", "w"))
    LOG = "/kaggle/working/results/respaldo-prueba.jsonl"; os.makedirs(os.path.dirname(LOG), exist_ok=True)
    a = dict(config="/tmp/tiny_64k.json", tokenizer="tokenizer/tokenizer.json", shards=MANIFIESTO,
             shard_roots="/kaggle/input", seq=256, micro=4, tokens_per_step=2048, steps=meta["step"] + 10, warmup=5,
             eval_every=10, eval_windows=2, log_every=5, save_final=True, tag=meta["tag"], out=LOG)
    S = Session(None, KaggleBackup(RESPALDO, tag=meta["tag"]), CUENTA, meta["tag"], "/tmp/respaldo-prueba",
                lambda r, d: train_command(a, r, d), LOG, no_lock=True, poll_s=0.5).run()
    filas = [json.loads(x) for x in open(LOG)] if os.path.exists(LOG) else []
    res = [r for r in filas if r["type"] == "resume"][-1:]
    lote = [r for r in filas if r["type"] == "resume_check"][-1:]
    for e in S["events"]:
        print(" ", e["kind"], {k: v for k, v in e.items() if k not in ("kind", "time", "by", "session")})
    pruebas = {
        f"siguió desde Kaggle en el paso {meta['step']}": S["resumed_from"] == {"where": "kaggle", "step": meta["step"]},
        "leyó el respaldo donde Kaggle lo montó, sin copiarlo": any(e["kind"] == "in_place" for e in S["events"]),
        "reanudación exacta: el estado": bool(res and res[0]["state_exact"]),
        "reanudación exacta: el primer lote": bool(lote and lote[0]["batch_exact"]),
        f"entrenó hasta el paso {meta['step'] + 10}": S["train_exit"] == 0 and S["last_local"] == f"step_{meta['step'] + 10:06d}",
        "respaldó en el otro dataset": any(e["kind"] == "backup" and e["dataset"] != slug for e in S["events"]),
    }
    print()
    for k, v in pruebas.items():
        print(" ", "✅" if v else "❌", k)
    print("\nRespaldo:", "✅ un turno puede seguir desde Kaggle" if all(pruebas.values())
          else "❌ revisar las condiciones y los eventos de arriba")'''),
    md("""## G0-1 · `g0-16k`

50 pasos del modelo real a 16K de contexto (un lote de 16,384 tokens por GPU), con los shards reales, sin guardar. Pasa si termina sin falta de memoria."""),
    code(r'''if MODO == "g0-16k":
    LOG = "/kaggle/working/results/g0-16k.jsonl"; os.makedirs(os.path.dirname(LOG), exist_ok=True)
    a = dict(TRONCO["args"], shards=MANIFIESTO, shard_roots="/kaggle/input", seq=16384, micro=1,
             tokens_per_step=32768, ramp_tokens=0, steps=50, warmup=10, eval_every=0, eval_windows=2,
             save_every_hours=0, save_final=False, keep=1, log_every=5, tag="g0-16k", out=LOG)
    p = subprocess.run(train_command(a, None, "/tmp/g0_16k"))
    fin = [json.loads(x) for x in open(LOG) if json.loads(x)["type"] == "end"][-1:] if os.path.exists(LOG) else []
    if p.returncode == 0 and fin and fin[0]["status"] == "done":
        f = fin[0]
        print(f"G0-1 ✅ 50 pasos a 16K sin falta de memoria: pico {f['peak_reserved_GiB']} GiB reservada, "
              f"{f['tokens_per_s_mean']} tokens/s")
    else:
        print("G0-1 ❌ no terminó: ver el log de arriba (¿out of memory?)")'''),
    md("""## Tronco · un turno

Entrena hasta ~11 h desde que empezó la sesión y deja ~45 min para la subida final, el respaldo (~11 GB a Kaggle) y el aplastado. Cada línea `[relevo]` es un evento: candado, de dónde siguió, cada subida, lo que se borra del disco, el respaldo y la entrega.

En `/tmp` nunca hay más de dos checkpoints a la vez (~23 GiB). El de partida se borra después del primer guardado nuevo, se guarda localmente solo el periódico más nuevo, y antes del respaldo solo queda el último."""),
    code(r'''if MODO == "tronco":
    LOG = f"/kaggle/working/results/tronco/{time.strftime('%Y%m%d-%H%M')}-{CUENTA.replace(' ', '-')}.jsonl"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    horas = round(11.0 - (time.time() - T_SESION) / 3600, 2)
    a = dict(TRONCO["args"], shards=MANIFIESTO, shard_roots="/kaggle/input", max_hours=horas, tag=TRONCO["tag"], out=LOG)
    tienda = None if SIN_CANDADO else HFStore(HF_REPO, os.environ.get("HF_TOKEN"))
    if not SIN_CANDADO:
        quien = Lock(tienda, CUENTA, "-").holder(tienda.head())
        if quien and not quien.get("released") and quien["expires"] > time.time():
            print(f"candado: lo tiene {quien['holder']} por {(quien['expires'] - time.time()) / 60:.0f} min más")
        else:
            print("candado: libre" + (f" (el último turno fue de {quien['holder']})" if quien else ""))
    s = Session(tienda, KaggleBackup(RESPALDO, tag=TRONCO["tag"]), CUENTA, TRONCO["tag"], "/tmp/relevo",
                lambda r, d: train_command(a, r, d), LOG, no_lock=SIN_CANDADO)
    if SIN_CANDADO and s.newest()[0] is None:   # sin Hugging Face, solo un respaldo dice dónde va el tronco
        raise SystemExit("⛔ SIN_CANDADO y ningún respaldo del tronco montado: no se entrena desde cero. "
                         "Adjunta ludwig2-respaldo-a y ludwig2-respaldo-b.")
    try:
        RESUMEN = s.run()
    except Busy as e:
        raise SystemExit(f"⛔ el relevo está ocupado: {e}. No entrenes: avisa por WhatsApp.")
    json.dump(RESUMEN, open(LOG.replace(".jsonl", ".relevo.json"), "w"), indent=1)
    if not SIN_CANDADO and any(e["kind"] == "release_failed" for e in RESUMEN["events"]):
        # Turno 1: la entrega falló justo después del aplastado (Hugging Face aún mostraba la cabeza vieja).
        # Se reintenta con la cabeza nueva, y solo si el candado sigue siendo de esta sesión.
        candado = Lock(tienda, CUENTA, RESUMEN["session"])
        for intento in range(1, 7):
            time.sleep(30)
            try:
                cabeza = tienda.head()
                if (candado.holder(cabeza) or {}).get("session") != RESUMEN["session"]:
                    print("candado: ya no es de este turno, no se toca")
                    break
                candado.release(cabeza)
                print(f"candado: soltado al reintentar (intento {intento})")
                break
            except Exception as e:  # noqa: BLE001 - si no se puede, vence solo a las 2 h de su última renovación
                print(f"candado: el intento {intento} falló ({type(e).__name__}: {str(e)[:120]})")
    filas = [json.loads(x) for x in open(LOG)]
    fin = ([r for r in filas if r["type"] == "end"] or [{}])[-1]
    res = [r for r in filas if r["type"] == "resume"][-1:]
    lote = [r for r in filas if r["type"] == "resume_check"][-1:]
    subidas = [e for e in RESUMEN["events"] if e["kind"] == "upload"]
    tps, pico = fin.get("tokens_per_s_last_hour"), fin.get("peak_reserved_GiB")
    print(f"\n{fin.get('status')}: paso {fin.get('steps_done')}, {fin.get('tokens', 0) / 1e9:.3f}B tokens, "
          f"{fin.get('hours_this_session')} h en esta sesión")
    print("Condiciones de la G0 en este turno:")
    print(f"  1. ≥6 h sin NaN, escala estable: {'✅' if (fin.get('hours_this_session') or 0) >= 6 and fin.get('status') != 'error' and (fin.get('loss_scale_last') or 0) >= 1 else '❌'} "
          f"({fin.get('loss_scale_skips')} saltos de FP16, escala final {fin.get('loss_scale_last')})")
    print(f"  2. ≥2,600 tokens/s en la última hora: {'✅' if tps and tps >= 2600 else '❌'} ({tps})")
    print(f"  3. ≤13.5 GiB reservada: {'✅' if pico and pico <= 13.5 else '❌'} ({pico})")
    print(f"  4. ≥2 subidas: {'✅' if len(subidas) >= 2 else '❌'} ({len(subidas)}); reanudación en otra cuenta: "
          + (f"{'✅' if res[0]['state_exact'] and lote and lote[0]['batch_exact'] else '❌'} (desde {RESUMEN['resumed_from']})"
             if res else "— (este turno empezó de cero)"))'''),
    md("""## Qué hacer con el resultado

1. **Del turno:**
   - De la salida (`results/tronco/`): el `.jsonl` del turno y su `.relevo.json`, con los eventos del relevo.
   - Van al repo en `results/tronco/`, con su fila en `results/README.md` y una entrada en la bitácora. El log también queda en `logs/` del repo de Hugging Face.
2. **WhatsApp:** "relevo libre, paso N". Si el resumen dice `lock_lost`, otra sesión tomó el candado después de que este venciera, y esta paró sola: avisa.
3. **La G0** está completa cuando:
   - el primer turno del tronco pasa las condiciones 1–4 (menos la reanudación);
   - el segundo turno, en otra cuenta, marca la reanudación ✅;
   - `g0-16k` dio ✅.

   Si algo falla, el checkpoint del primer turno sigue sirviendo salvo que lo que falló sea el entrenamiento mismo (NaN, escala)."""),
]

for nb, name in [(t01, "t0_1_formato_mtp_tokenizador.ipynb"), (t02, "t0_2_velocidad_memoria_2xT4.ipynb"),
                 (t02v2, "t0_2v2_afmoe_2xT4.ipynb"), (t03, "t0_3_t2_lr_2xT4.ipynb"), (t1, "t1_shards_cpu.ipynb"),
                 (tronco, "tronco_relevo_2xT4.ipynb")]:
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:02d}"          # stable ids: regenerating gives an identical file
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote notebooks/" + name)
