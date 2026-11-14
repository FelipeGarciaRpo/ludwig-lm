# ══════════════════════════════════════════════════════════════════
# TOKENIZADOR 64K DE LUDWIG 2 — notebook de Kaggle (CPU: no gasta cuota de GPU)
# ──────────────────────────────────────────────────────────────────
# Deja en /kaggle/working/tokenizador_64k/:
#   tokenizer.json, tokenizer_config.json, special_tokens_map.json   el tokenizador (chat y servir)
#   cargar_tokenizador.py   para_datos(): el texto crudo NUNCA produce tokens de control
#                           para_chat():  el normal, con <|im_start|>, <think>, FIM…
#   tokenizer_meta.json     receta, mezcla, archivos usados, licencias, sha1, versiones y métricas
#   compresion.csv          bytes por token por fuente: el nuevo contra Ludwig 1, SmolLM2, OLMoE, Qwen3 y Llama 3
#   mezclas.csv             la misma medida con otras mezclas de la muestra (más o menos español)
#   reporte.txt             lo mismo que imprime al final
#   muestra/                el texto con que se entrenó y los vals (para repetir sin bajar nada)
#                           ⚠ es texto crudo de los datasets: no publiques el output del notebook con esta carpeta
#
# v1.2 (13 nov 2026): las fuentes pasan a ser las de la mezcla v3 del tronco de Ludwig 2.
#   · Inglés: FineWeb-Edu y DCLM sin duplicados, FinePhrase (sus 4 formatos), Wikipedia y Cosmopedia.
#   · Español: FineWeb2-HQ (nativo), TransWeb-Edu (traducido) y Wikipedia.
#   · Código: Stack-Edu SOLO con licencia permisiva. Stack-Edu trae identificadores, no el contenido: cada
#     archivo se baja de Software Heritage (unos 120 mil en la corrida de verdad). HTML y CSS salen de
#     The Stack v1, que es todo permisivo. Salen C#, Java, Go, Rust, C y C++.
#   · Matemática: FineMath 4+ e InfiWebMath 4+. Razonamiento: OpenMathReasoning.
#   · Los archivos se leen con pyarrow y solo las columnas que hacen falta (FineWeb2-HQ trae los embeddings
#     de cada documento: sin esto se bajarían enteros).
#   · Las cadenas de los tokens de control ("<think>", "<|im_start|>"…) se quitan del texto de la muestra:
#     en el entrenamiento las pone el lector como ids, así que no vale la pena gastar merges en ellas.
#   · Compara otras mezclas de la muestra en el tamaño chico del barrido (proporcional al tronco, más
#     español, como la v2): muestra cuánto cuesta en inglés y en código darle más peso al español.
#   · El promedio "del tronco" pesa cada dominio por su parte de los TOKENS del tronco (36/24/28/10/2).
#   · Si una fuente tiene un solo archivo (TypeScript en Stack-Edu), su val es el último 10% de las filas.
#   · Probada el 13 nov 2026 en modo rápido contra las fuentes reales, dos veces: abrieron las 23, Software
#     Heritage entregó 1,730 archivos sin fallos y todas las pruebas salieron con ✅. En un PC tomó 8 min con
#     buena conexión y 43 min con una lenta: DCLM y Stack Overflow son las fuentes que más demoran.
#     En Kaggle (14 nov) la prueba rápida tomó 7 min y dio el mismo tokenizador, byte por byte, que en el PC.
#   · Freno de memoria en el barrido: si el tamaño siguiente no va a caber en la RAM, se queda con el anterior
#     en vez de tumbar la corrida.
# v1.1 (10 nov 2026): submuestras anidadas y parejas en el barrido; promedio armónico de la mezcla.
#
# La receta (no cambia):
#   · BPE a nivel de byte de 65,536 tokens: 256 especiales (ids 0–255) + 256 bytes + 65,024 merges.
#     De los 256 especiales, 16 tienen nombre y 240 quedan de reserva (<|reserved_0|> … <|reserved_239|>).
#   · Los 256 bytes siempre en el vocabulario: ningún carácter se pierde (lo que le faltó a Ludwig 1).
#   · Regex de Qwen2: dígitos uno por uno; llama.cpp ya lo soporta (pre-tokenizador "qwen2").
#   · Sin normalizador: el texto sale idéntico, byte por byte (importa en código).
#   · Se entrena con documentos enteros, no línea por línea: así aprende "\n\n" y la indentación.
#   · El val de cada fuente nunca entra al tokenizador.
#
# ANTES DE CORRER
#   Settings → Accelerator: None (CPU) · Internet: On
#   Add-ons → Secrets → HF_TOKEN (la cuenta que aceptó la licencia de bigcode/the-stack-dedup). Solo hace
#   falta para HTML y CSS: sin token esas dos quedan fuera y su peso pasa al resto del código.
#   Opcional: Add Input → el output de tu notebook de datos de Ludwig 1, para compararlo.
# CÓMO CORRER
#   a) PRUEBA_RAPIDA = True → Run All (≈15–25 min). Revisa que todo salga con ✅.
#   b) PRUEBA_RAPIDA = False → Save Version → "Save & Run All (Commit)" (≈3–5 h en segundo plano).
# ══════════════════════════════════════════════════════════════════
PRUEBA_RAPIDA = True              # ← True para probar; False para la corrida de verdad
TAMANOS_GB    = [0.3, 0.6, 1.2]   # barrido del tamaño de la muestra; el último es el tokenizador final
VOCAB         = 65_536
N_ESPECIALES  = 256
SEMILLA       = 2026
MAX_CHARS_DOC = 20_000            # cada documento se corta aquí (los gigantes no aportan merges nuevos)
DOCS_VAL      = 300               # docs de val por fuente, para medir compresión
ARCHIVOS_POR_FUENTE = 12          # de cuántos archivos distintos sale la muestra de cada fuente
MB_SIN_USO    = 100               # texto con que se buscan tokens que nunca salen
HORAS_MAX     = 11.0
COMPARAR_MEZCLAS    = True        # entrena 3 tokenizadores chicos más, con otras mezclas, para compararlas
USAR_TRANSWEB       = True        # False si no se confirma quién tradujo TransWeb-Edu: su peso pasa al español nativo
CODIGO_DE_THE_STACK = False       # True: todo el código sale de The Stack v1 (plan B si Software Heritage no responde)
HILOS_SWH     = 32                # bajadas simultáneas de Software Heritage

import os, sys, re, json, time, gzip, random, hashlib, shutil, platform, subprocess, zlib
import urllib.request, urllib.error
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
def log(*a): print(*a, flush=True)
def horas(): return (time.time() - T0) / 3600

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")   # lecturas lentas de Hugging Face: esperar antes de reintentar
TEST = os.environ.get("TOK_TEST") == "1"          # prueba local, sin internet (datos sintéticos)
if os.environ.get("TOK_RAPIDA") in ("0", "1"): PRUEBA_RAPIDA = os.environ["TOK_RAPIDA"] == "1"
if TEST: PRUEBA_RAPIDA = True
if PRUEBA_RAPIDA:
    TAMANOS_GB, DOCS_VAL, ARCHIVOS_POR_FUENTE, MB_SIN_USO = [0.01, 0.02], 40, 3, 5
if TEST:
    TAMANOS_GB, MB_SIN_USO = [0.002, 0.004], 1
    VOCAB = int(os.environ.get("TOK_VOCAB", "1024"))

def _pip(*paquetes): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *paquetes], check=True)
try:
    import tokenizers
except ImportError:
    _pip("tokenizers"); import tokenizers
if tuple(int(x) for x in tokenizers.__version__.split(".")[:2]) < (0, 19):
    _pip("-U", "tokenizers")
    raise RuntimeError("Actualicé tokenizers (hace falta ≥ 0.19). Reinicia la sesión (Run → Restart) y corre de nuevo.")
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
if not TEST:
    try:
        import pyarrow, huggingface_hub
    except ImportError:
        _pip("pyarrow", "huggingface_hub"); import pyarrow, huggingface_hub
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
try:
    import psutil
except ImportError:
    psutil = None

ENTRADA = Path(os.environ.get("TOK_INPUT", "/kaggle/input"))
SALIDA  = Path(os.environ.get("TOK_OUTPUT", "/kaggle/working")) / "tokenizador_64k"
MUESTRA = SALIDA / "muestra"
shutil.rmtree(SALIDA, ignore_errors=True); MUESTRA.mkdir(parents=True)
log(f"tokenizers {tokenizers.__version__} | PRUEBA_RAPIDA={PRUEBA_RAPIDA} | tamaños {TAMANOS_GB} GB | vocab {VOCAB:,}")

# ─── 1) token de Hugging Face (nunca se imprime) ───
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and not TEST:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
if not TEST:
    log("✅ HF_TOKEN encontrado" if HF_TOKEN else
        "⚠ Sin HF_TOKEN: HTML y CSS (the-stack-dedup) no van a abrir (Add-ons → Secrets → HF_TOKEN).")

# ─── 2) receta ───
NOMBRADOS = [          # (texto, especial). especial=False: se ve al decodificar, como en Qwen3
    ("<|endoftext|>", True),                  # id 0: fin de documento (EOS del preentrenamiento)
    ("<|pad|>", True),                        # id 1: relleno
    ("<|im_start|>", True), ("<|im_end|>", True),                      # ids 2–3: chat (formato ChatML)
    ("<think>", False), ("</think>", False),                           # ids 4–5: razonamiento
    ("<tool_call>", False), ("</tool_call>", False),                   # ids 6–9: herramientas
    ("<tool_response>", False), ("</tool_response>", False),
    ("<|fim_prefix|>", True), ("<|fim_middle|>", True), ("<|fim_suffix|>", True), ("<|fim_pad|>", True),  # ids 10–13
    ("<|repo_name|>", True), ("<|file_sep|>", True),                   # ids 14–15: código a nivel de repositorio
]
ESPECIALES = [t for t, _ in NOMBRADOS] + [f"<|reserved_{i}|>" for i in range(N_ESPECIALES - len(NOMBRADOS))]
VISIBLES = {t for t, e in NOMBRADOS if not e}
REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
assert VOCAB <= 65_536, "con más de 65,536 tokens los ids ya no caben en uint16"
# En el texto de la muestra, las cadenas de control se cambian por un salto de línea (ver la nota de la v1.2)
CONTROL = re.compile("|".join(re.escape(t) for t, _ in NOMBRADOS))

# Cada fuente: de dónde sale, qué campo es el texto, su dominio y su peso en la muestra (en bytes de texto).
def hf(repo, prefijo, campo, dominio, peso, licencia, **extra):
    return dict(tipo="hf", repo=repo, prefijo=prefijo, campo=campo, dominio=dominio, peso=peso, licencia=licencia, **extra)

SE, SD = "HuggingFaceTB/stack-edu", "bigcode/the-stack-dedup"
def stack_edu(carpeta, peso, lineas=True):
    """Stack-Edu: código con puntaje educativo ≥ 3. Solo entran los archivos con licencia permisiva."""
    return dict(tipo="stack-edu", repo=SE, prefijo=f"{carpeta}/", campo="content", dominio="código", peso=peso,
                codigo=True, lineas=lineas, columnas=["blob_id", "license_type", "length_bytes", "src_encoding"],
                licencia="la de cada archivo (solo license_type = permissive); contenido bajado de Software Heritage")
def stack_v1(lenguaje, peso, lineas=True):
    """The Stack v1 sin duplicados: solo repositorios con licencia permisiva; trae el contenido."""
    return dict(tipo="hf", repo=SD, prefijo=f"data/{lenguaje}/", campo="content", dominio="código", peso=peso,
                codigo=True, lineas=lineas,
                columnas=["content", "max_line_length", "avg_line_length", "alphanum_fraction"],
                licencia="licencia permisiva de cada archivo + términos de uso de The Stack")
def finephrase(formato, peso):
    """FinePhrase: páginas de FineWeb-Edu reescritas por SmolLM2-1.7B. La reescritura está en rollout_results;
    la columna `text` es la página ORIGINAL y no se usa."""
    return hf("HuggingFaceFW/finephrase", f"{formato}/", "rollout_results", "inglés", peso,
              "ODC-By (reescrito con SmolLM2-1.7B-Instruct, Apache 2.0)", extraer="finephrase",
              columnas=["rollout_results"], min_chars=200)   # hay reescrituras de una sola línea: fuera
codigo = stack_v1 if CODIGO_DE_THE_STACK else stack_edu
def nombre_de(lenguaje):                     # las carpetas de The Stack v1 van en minúscula
    return lenguaje.lower() if CODIGO_DE_THE_STACK else lenguaje

CC = "ODC-By + términos de uso de Common Crawl"
FUENTES = {
    # ── inglés, 34% ──
    "fineweb-edu":       hf("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup/", "text", "inglés", 0.130, CC),
    "dclm":              hf("Zyphra/dclm-dedup", "data/dclm-dedup/", "text", "inglés", 0.105,
                            "CC-BY-4.0 (la ficha de DCLM dice: solo para investigación)"),
    "finephrase-faq":    finephrase("faq", 0.014), "finephrase-tutorial": finephrase("tutorial", 0.014),
    "finephrase-math":   finephrase("math", 0.014), "finephrase-table":   finephrase("table", 0.014),
    "wiki-en":           hf("wikimedia/wikipedia", "20231101.en/", "text", "inglés", 0.029, "CC-BY-SA"),
    "cosmopedia":        hf("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", "text", "inglés", 0.020,
                            "ODC-By (generado con Mixtral-8x7B, Apache 2.0)"),
    # ── español, 26% ──
    "fineweb2-hq-es":    hf("epfml/FineWeb2-HQ", "spa_Latn/", "text", "español", 0.140, CC),
    "transweb-es":       hf("britllm/TransWeb-Edu-Spanish", "es_chunk", "text_es", "español", 0.090,
                            "ODC-By (traducción automática; falta confirmar con qué modelo)", ext=".jsonl"),
    "wiki-es":           hf("wikimedia/wikipedia", "20231101.es/", "text", "español", 0.030, "CC-BY-SA"),
    # ── código, 26% ──
    "python":            codigo(nombre_de("Python"), 0.104),
    "javascript":        codigo(nombre_de("JavaScript"), 0.068),
    "typescript":        codigo(nombre_de("TypeScript"), 0.036),
    "markdown":          codigo(nombre_de("Markdown"), 0.014, lineas=False),
    "sql":               codigo(nombre_de("SQL"), 0.005),
    "shell":             codigo(nombre_de("Shell"), 0.005),
    "stackoverflow":     hf("mikex86/stackoverflow-posts", "stackoverflow-posts-", "Body", "código", 0.015,
                            "CC-BY-SA (volcado de 2023, anterior al cambio de condiciones de julio de 2024)",
                            extraer="stackoverflow", columnas=["Title", "Body", "Score"]),
    "html":              stack_v1("html", 0.007),
    "css":               stack_v1("css", 0.006),
    # ── matemática, 10% ──
    "finemath-4plus":    hf("HuggingFaceTB/finemath", "finemath-4plus/", "text", "matemática", 0.055, CC),
    "infiwebmath-4plus": hf("HuggingFaceTB/finemath", "infiwebmath-4plus/", "text", "matemática", 0.045, CC),
    # ── razonamiento, 4% ──
    "openmathreasoning": hf("nvidia/OpenMathReasoning", "data/cot-", "generated_solution", "razonamiento", 0.040,
                            "CC-BY-4.0 (trazas de DeepSeek-R1 y QwQ-32B)", extraer="problema+solucion",
                            columnas=["problem", "generated_solution"]),
}
assert abs(sum(F["peso"] for F in FUENTES.values()) - 1) < 1e-9
# No se muestrean (su texto no trae tipos de token nuevos y su fuente aún no está elegida):
# StackExchange general, los notebooks y el texto paralelo inglés–español.
NO_MUESTREADAS = ["StackExchange general", "notebooks", "texto paralelo inglés–español"]
OBLIGATORIAS = {"fineweb-edu", "dclm", "fineweb2-hq-es", "python", "javascript", "typescript", "finemath-4plus"}
# Parte de los TOKENS del tronco que tiene cada dominio (mezcla v3). Con esto se calcula el promedio "del tronco".
TRONCO = {"inglés": 0.36, "español": 0.24, "código": 0.28, "matemática": 0.10, "razonamiento": 0.02}
# Otras mezclas de la muestra, por dominio (dentro de cada dominio se reparte igual que en la final)
MEZCLAS = {
    "proporcional al tronco": {"inglés": 0.41, "español": 0.25, "código": 0.23, "matemática": 0.09, "razonamiento": 0.02},
    "más español":            {"inglés": 0.30, "español": 0.30, "código": 0.26, "matemática": 0.10, "razonamiento": 0.04},
    "como la v2":             {"inglés": 0.53, "español": 0.16, "código": 0.21, "matemática": 0.08, "razonamiento": 0.02},
}
REFERENCIAS = {"SmolLM2 (49K)": "HuggingFaceTB/SmolLM2-360M", "OLMoE (50K)": "allenai/OLMoE-1B-7B-0125",
               "Qwen3 (151K)": "Qwen/Qwen3-0.6B", "Llama 3 (128K)": "meta-llama/Llama-3.2-1B"}

# ─── 3) lectura ───
def texto_de(ex, F):
    """El texto de un documento según la fuente; "" si el documento no sirve."""
    if "_texto" in ex: return ex["_texto"]                     # prueba local
    modo = F.get("extraer")
    if modo == "finephrase":                                   # la primera reescritura que terminó bien
        for r in ex.get("rollout_results") or []:
            if isinstance(r, dict) and r.get("finish_reason") == "stop" and (r.get("text") or "").strip():
                return r["text"]
        return ""
    if modo == "problema+solucion":
        p, s = ex.get("problem") or "", ex.get("generated_solution") or ""
        return f"{p}\n\n{s}" if p and s else ""
    if modo == "stackoverflow":                                # preguntas y respuestas con al menos un voto
        if (ex.get("Score") or 0) < 1: return ""
        t, b = ex.get("Title") or "", ex.get("Body") or ""
        return f"{t}\n\n{b}" if t else b
    v = ex.get(F["campo"])
    return v if isinstance(v, str) else ""

def limpiar(t):
    """Corta el documento y cambia las cadenas de control por un salto de línea."""
    return CONTROL.sub("\n", t[:MAX_CHARS_DOC]) if t else ""

AUTOGEN = ("auto-generated", "autogenerated", "automatically generated", "do not edit")
def pasa(ex, texto, F):
    if len(texto.strip()) < F.get("min_chars", 50): return False
    if F.get("codigo"):
        if len(texto) > 100_000: return False
        if F.get("lineas", True):
            lmax, lprom, alfa = ex.get("max_line_length"), ex.get("avg_line_length"), ex.get("alphanum_fraction")
            if lmax is None or lprom is None or alfa is None:  # Stack-Edu no trae estas columnas: se calculan
                lineas = texto.splitlines() or [""]
                lmax, lprom = max(map(len, lineas)), sum(map(len, lineas)) / len(lineas)
                alfa = sum(map(str.isalnum, texto)) / max(1, len(texto))
            if lmax > 1000 or lprom > 100 or alfa < 0.25: return False
        cabeza = "\n".join(texto[:2000].splitlines()[:5]).lower()
        if any(k in cabeza for k in AUTOGEN): return False
    return True

def columnas(F): return F.get("columnas") or [F["campo"]]

SWH = dict(pedidos=0, fallos=0)                                # bajadas de Software Heritage
if TEST:                                                       # fuentes sintéticas para probar sin internet
    PAL = {"inglés": ("the of and to in is was for on that with as by at from which this are be have it not "
                      "model learning energy history science water language computer people world").split(),
           "español": ("el la de que y en los se del las un por con no una su para es al lo como más pero sus "
                       "año niño ñandú corazón canción también qué día país según así después mundo").split()}
    def _doc(dominio, f, i):
        r = random.Random(f"{f}-{i}")
        if dominio == "código":
            return "\n\n".join(f"def {r.choice(['suma', 'cargar_datos', 'getUserName', 'parse_json'])}{r.randint(0, 99)}"
                               f"(x, y={r.randint(0, 9)}):\n    if x > {r.randint(0, 999)}:\n        return x + y"
                               f"  # comentario\n    return None" for _ in range(r.randint(2, 12)))
        if dominio == "matemática":
            return " ".join(f"Sea $x^{r.randint(2, 9)} + {r.randint(1, 99)}x = {r.randint(0, 999)}$, entonces "
                            f"\\frac{{a}}{{b}} = {r.random():.4f}." for _ in range(r.randint(2, 10)))
        if dominio == "razonamiento":
            return f"<think>\nSi x = {r.randint(1, 99)}, entonces 2x = {r.randint(2, 198)}.\n</think>\n\nLa respuesta es {r.randint(0, 999)}. " * r.randint(2, 6)
        pal = PAL["español" if dominio == "español" else "inglés"]
        return "\n\n".join(" ".join(r.choice(pal) for _ in range(r.randint(20, 120))).capitalize() + "."
                           for _ in range(r.randint(1, 5))) + (" ¿Qué tal? ¡Olé! 😀" if dominio == "español" else " It's ok.")
    def listar(F): return [f"{F['prefijo']}archivo-{k:03d}.parquet" for k in range(6)]
    def n_filas(F, f): return 400
    def _leer(F, f, saltar, hasta):
        for i in range(saltar, min(400, hasta if hasta is not None else 400)):
            yield i, {"_texto": _doc(F["dominio"], f, i)}
else:
    _fs, _listas = None, {}
    def fs():
        global _fs
        if _fs is None: _fs = HfFileSystem(token=HF_TOKEN)
        return _fs

    def listar(F):
        """Los archivos de la fuente, ordenados. El prefijo puede ser una carpeta o el comienzo de un nombre."""
        repo, prefijo, ext = F["repo"], F["prefijo"], F.get("ext", ".parquet")
        if (repo, prefijo) not in _listas:
            carpeta = prefijo.rsplit("/", 1)[0] if "/" in prefijo else ""
            api = HfApi()
            try:
                arbol = api.list_repo_tree(repo, path_in_repo=carpeta or None, repo_type="dataset",
                                           recursive=True, token=HF_TOKEN)
                rutas = [x.path for x in arbol if type(x).__name__ == "RepoFile"]
            except Exception:
                rutas = api.list_repo_files(repo, repo_type="dataset", token=HF_TOKEN)
            _listas[(repo, prefijo)] = sorted(r for r in rutas if r.startswith(prefijo) and r.endswith(ext))
        return _listas[(repo, prefijo)]

    def n_filas(F, f):
        with fs().open(f"datasets/{F['repo']}/{f}", "rb") as fh:
            return pq.ParquetFile(fh).metadata.num_rows

    def leer_parquet(repo, f, cols, saltar=0, hasta=None, lote=64):
        """(posición, fila) desde la fila `saltar`. Salta grupos de filas enteros sin bajarlos y lee solo `cols`."""
        with fs().open(f"datasets/{repo}/{f}", "rb") as fh:
            pf = pq.ParquetFile(fh, buffer_size=4 << 20)
            md, inicios = pf.metadata, [0]
            for g in range(md.num_row_groups): inicios.append(inicios[-1] + md.row_group(g).num_rows)
            g0 = max([g for g in range(md.num_row_groups) if inicios[g] <= saltar], default=0)
            cols = [c for c in cols if c in pf.schema_arrow.names] or None
            pos = inicios[g0]
            for tanda in pf.iter_batches(batch_size=lote, row_groups=list(range(g0, md.num_row_groups)), columns=cols):
                for fila in tanda.to_pylist():
                    if hasta is not None and pos >= hasta: return
                    if pos >= saltar: yield pos, fila
                    pos += 1

    def leer_jsonl(repo, f, saltar=0, hasta=None):
        with fs().open(f"datasets/{repo}/{f}", "rb") as fh:
            for pos, linea in enumerate(fh):
                if hasta is not None and pos >= hasta: return
                if pos < saltar: continue
                try: yield pos, json.loads(linea)
                except Exception: continue

    _pool = None
    def bajar_swh(fila, intentos=3):
        """El contenido de un archivo de Stack-Edu desde el bucket público de Software Heritage; None si no está."""
        url = f"https://softwareheritage.s3.amazonaws.com/content/{fila['blob_id']}"
        for k in range(intentos):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    datos = gzip.decompress(r.read())
                try: return datos.decode(fila.get("src_encoding") or "utf-8", errors="ignore")
                except LookupError: return datos.decode("utf-8", errors="ignore")
            except urllib.error.HTTPError as e:
                if e.code in (403, 404): return None
            except Exception:
                pass
            time.sleep(1 + 2 * k)
        return None

    def leer_stack_edu(repo, f, cols, saltar=0, hasta=None):
        """Filas de Stack-Edu con licencia permisiva y su contenido, bajado en tandas de 64 archivos."""
        global _pool
        if _pool is None: _pool = ThreadPoolExecutor(HILOS_SWH)
        def bajar(tanda):
            for (pos, fila), texto in zip(tanda, _pool.map(bajar_swh, [x for _, x in tanda])):
                SWH["pedidos"] += 1
                if texto is None: SWH["fallos"] += 1; continue
                yield pos, dict(fila, content=texto)
        tanda = []
        for pos, fila in leer_parquet(repo, f, cols, saltar, hasta, lote=512):
            if fila.get("license_type") != "permissive": continue
            if not 50 <= (fila.get("length_bytes") or 0) <= 100_000: continue
            tanda.append((pos, fila))
            if len(tanda) >= 64:
                yield from bajar(tanda); tanda = []
        if tanda: yield from bajar(tanda)

    def _leer(F, f, saltar, hasta):
        if F["tipo"] == "stack-edu": return leer_stack_edu(F["repo"], f, columnas(F), saltar, hasta)
        if f.endswith(".jsonl"): return leer_jsonl(F["repo"], f, saltar, hasta)
        return leer_parquet(F["repo"], f, columnas(F), saltar, hasta)

def leer(F, f, saltar=0, hasta=None, reintentos=4):
    """(posición, fila) del archivo f desde la fila `saltar`. Si la conexión se cae, reabre y sigue donde iba."""
    sigue, fallos = saltar, 0
    while True:
        try:
            for pos, fila in _leer(F, f, sigue, hasta):
                sigue = pos + 1
                yield pos, fila
            return
        except Exception as e:
            fallos += 1
            if fallos > reintentos:
                log(f"   ⚠ salto {f} ({type(e).__name__}: {str(e)[:120]})")
                return
            log(f"   ⚠ se cayó la lectura de {f} ({type(e).__name__}); reintento {fallos}/{reintentos}…")
            time.sleep(5 * fallos)

# ─── 4) accesos: si una fuente no abre, su peso pasa a las de su mismo dominio ───
def quitar(nombre, motivo):
    F = FUENTES.pop(nombre)
    destino = [m for m, G in FUENTES.items() if G["dominio"] == F["dominio"]] or list(FUENTES)
    tot = sum(FUENTES[m]["peso"] for m in destino)
    for m in destino: FUENTES[m]["peso"] += F["peso"] * FUENTES[m]["peso"] / tot
    log(f"   → {nombre} queda fuera ({motivo}); su {F['peso']:.1%} pasa a: {', '.join(destino)}")

def probar(nombre, F):
    try:
        archivos = listar(F)
        assert archivos, f"no encontré archivos {F.get('ext', '.parquet')} con ese prefijo"
        ex = next(iter(leer(F, archivos[0], reintentos=1)), None)
        assert ex is not None, "no pude leer ningún documento"
        log(f"   ✅ {nombre:<19} {F['repo']}/{F['prefijo']}  ({len(archivos)} archivos)")
        return True
    except Exception as e:
        log(f"   ❌ {nombre:<19} {F['repo']}/{F['prefijo']}: {type(e).__name__}: {str(e)[:200]}")
        return False

log("\nAccesos:")
fallan = []
if not USAR_TRANSWEB:
    quitar("transweb-es", "USAR_TRANSWEB = False"); fallan.append("transweb-es (desactivada)")
for n, F in list(FUENTES.items()):
    if probar(n, F): continue
    if n in OBLIGATORIAS:
        raise RuntimeError(f"No abre {n} y sin esa fuente no hay tokenizador. ¿Internet en On? Si es código de "
                           f"Stack-Edu, puede ser Software Heritage: prueba de nuevo o pon CODIGO_DE_THE_STACK = True.")
    quitar(n, "no abre"); fallan.append(n)

# ─── 5) muestra: un pedazo de varios archivos de cada fuente; el val sale de un archivo aparte ───
def muestrear(nombre, F, meta_bytes):
    archivos, limite = listar(F), {}
    if len(archivos) >= 2:
        archivo_val, train, val_desde = archivos[-1], archivos[:-1], 0
    else:                                            # un solo archivo: su último 10% de filas es el val
        assert archivos[0].endswith(".parquet"), f"{nombre}: un solo archivo y no es parquet"
        corte = int(n_filas(F, archivos[0]) * 0.9)
        archivo_val, train, val_desde, limite[archivos[0]] = archivos[0], archivos, corte, corte
    n_val = 0
    with open(MUESTRA / f"val_{nombre}.jsonl", "w", encoding="utf-8") as fv:
        for _, ex in leer(F, archivo_val, saltar=val_desde):
            t = limpiar(texto_de(ex, F))
            if not pasa(ex, t, F): continue
            fv.write(json.dumps(t, ensure_ascii=False) + "\n"); n_val += 1
            if n_val >= DOCS_VAL: break
    orden = train[:]; random.Random(f"{SEMILLA}-{nombre}").shuffle(orden)
    orden = orden[:ARCHIVOS_POR_FUENTE]
    st = dict(bytes=0, docs=0, leidos={})
    cuota = meta_bytes / len(orden)
    with open(MUESTRA / f"{nombre}.jsonl", "w", encoding="utf-8") as ft:
        for ronda in range(4):                       # rondas de reserva por si un archivo se agota
            antes = st["bytes"]
            for f in orden:
                if st["bytes"] >= meta_bytes: break
                objetivo, vistos = min(meta_bytes, st["bytes"] + cuota), st["leidos"].get(f, 0)
                for pos, ex in leer(F, f, saltar=vistos, hasta=limite.get(f)):
                    vistos = pos + 1
                    t = limpiar(texto_de(ex, F))
                    if pasa(ex, t, F):
                        ft.write(json.dumps(t, ensure_ascii=False) + "\n")
                        st["bytes"] += len(t.encode("utf-8")); st["docs"] += 1
                    if st["bytes"] >= objetivo: break
                st["leidos"][f] = vistos
            if st["bytes"] >= meta_bytes or st["bytes"] == antes: break
    return dict(bytes=st["bytes"], docs=st["docs"], docs_val=n_val, archivo_val=archivo_val,
                archivos_usados=sorted(st["leidos"]), archivos_train_disponibles=len(train))

MAX_GB = max(TAMANOS_GB)
log(f"\nMuestra ({MAX_GB} GB de texto; val: {DOCS_VAL} docs por fuente):")
STATS = {}
for nombre, F in list(FUENTES.items()):
    t = time.time()
    try:
        STATS[nombre] = muestrear(nombre, F, F["peso"] * MAX_GB * 1e9)
        assert STATS[nombre]["docs"] and STATS[nombre]["docs_val"], "no salió ningún documento"
    except Exception as e:
        if nombre in OBLIGATORIAS: raise
        log(f"   ⚠ {nombre}: falló al muestrear ({type(e).__name__}: {str(e)[:150]})")
        STATS.pop(nombre, None); quitar(nombre, "falló al muestrear"); fallan.append(nombre); continue
    s = STATS[nombre]
    log(f"   {nombre:<19} {s['bytes']/1e6:>8.1f} MB de {F['peso'] * MAX_GB * 1e3:>7.1f} | {s['docs']:>8,} docs | "
        f"val {s['docs_val']} docs de {s['archivo_val'].split('/')[-1]} | {time.time() - t:.0f} s")
    if s["bytes"] < 0.5 * F["peso"] * MAX_GB * 1e9:
        log(f"   ⚠ {nombre}: llegó a menos de la mitad de su cuota")
if SWH["pedidos"]:
    log(f"   Software Heritage: {SWH['pedidos']:,} archivos pedidos, {SWH['fallos']:,} no bajaron "
        f"({SWH['fallos'] / SWH['pedidos']:.1%})")

def pesos_de(dominios):
    """Pesos por fuente de una mezcla dada por dominio: dentro del dominio se reparte como en la final."""
    total = Counter()
    for F in FUENTES.values(): total[F["dominio"]] += F["peso"]
    return {n: dominios.get(F["dominio"], 0.0) * F["peso"] / total[F["dominio"]] for n, F in FUENTES.items()}

def docs_muestra(gb, pesos=None):
    """Docs enteros de la muestra, en la proporción de la mezcla (la final, o la que se pase en `pesos`).
    Los tamaños chicos son subconjuntos anidados y parejos de los grandes: cada doc entra según un hash fijo,
    así que el de 0.3 GB ve los mismos archivos que el de 1.2 GB (no solo los primeros) y el barrido mide
    el efecto del tamaño, no el de la diversidad."""
    pesos = pesos or {n: F["peso"] for n, F in FUENTES.items()}
    total = sum(pesos.values())
    for nombre in FUENTES:
        frac = min(1.0, pesos[nombre] / total * gb * 1e9 / max(1, STATS[nombre]["bytes"]))
        umbral = frac * 2**32
        with open(MUESTRA / f"{nombre}.jsonl", encoding="utf-8") as fh:
            for k, linea in enumerate(fh):
                if frac < 1.0 and zlib.crc32(f"{SEMILLA}-{nombre}-{k}".encode()) >= umbral:
                    continue
                yield json.loads(linea)

VAL = {}
for nombre in FUENTES:
    with open(MUESTRA / f"val_{nombre}.jsonl", encoding="utf-8") as fh:
        VAL[nombre] = [json.loads(l) for l in fh]

# ─── 6) entrenamiento ───
def ram_pico_gb():
    """Pico de memoria del proceso según el sistema operativo (cuenta lo que usa Rust al entrenar).
    Es el máximo desde que arrancó el notebook: como los tamaños van de menor a mayor, es el del último."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: KB → GB
    except Exception:
        return float("nan")
RAM_TOTAL = psutil.virtual_memory().total / 1e9 if psutil else float("nan")

def entrenar(gb, pesos=None):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, special_tokens=ESPECIALES,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t = time.time()
    tk.train_from_iterator(docs_muestra(gb, pesos), trainer=tr)
    # versión de chat y servir: <think> y los de herramientas no son "special" (se ven al decodificar)
    j = json.loads(tk.to_str())
    for a in j["added_tokens"]: a["special"] = a["content"] not in VISIBLES
    return Tokenizer.from_str(json.dumps(j, ensure_ascii=False)), time.time() - t, ram_pico_gb()

def para_datos(tk):
    """Para tokenizar texto crudo: todo token de control se trata como texto normal."""
    j = json.loads(tk.to_str())
    for a in j["added_tokens"]: a["special"] = True
    t = Tokenizer.from_str(json.dumps(j, ensure_ascii=False)); t.encode_special_tokens = True
    return t

def bytes_por_token(tk, textos):
    if not textos: return float("nan")
    n = sum(len(e.ids) for e in tk.encode_batch(textos, add_special_tokens=False))
    return sum(len(t.encode("utf-8")) for t in textos) / max(1, n)

def _validos(pares): return [(p, v) for p, v in pares if v == v and v > 0]

def promedio_mezcla(fila):
    """Bytes por token de la MUESTRA = bytes totales / tokens totales.
    Con pesos en bytes, eso es el promedio armónico ponderado (el promedio simple sale inflado)."""
    pares = _validos((F["peso"], fila.get(n, float("nan"))) for n, F in FUENTES.items())
    return sum(p for p, _ in pares) / max(1e-12, sum(p / v for p, v in pares))

def por_dominio(fila):
    """Bytes por token de cada dominio (armónico, con los pesos en bytes de sus fuentes)."""
    out = {}
    for d in TRONCO:
        pares = _validos((F["peso"], fila.get(n, float("nan"))) for n, F in FUENTES.items() if F["dominio"] == d)
        if pares: out[d] = sum(p for p, _ in pares) / sum(p / v for p, v in pares)
    return out

def promedio_tronco(fila):
    """Bytes de texto que cubre, en promedio, un token del TRONCO. La mezcla del tronco está definida en tokens,
    así que cada dominio pesa por su parte de los tokens (promedio simple ponderado). Más es mejor: con los
    mismos 22.5B tokens el modelo ve más texto."""
    d = por_dominio(fila)
    return sum(TRONCO[k] * v for k, v in d.items()) / max(1e-12, sum(TRONCO[k] for k in d))

log(f"\nEntrenamiento (RAM total {RAM_TOTAL:.0f} GB):")
BARRIDO, final = [], None
for gb in TAMANOS_GB:
    if horas() > HORAS_MAX - 1.5 and final is not None:
        log(f"   ⏱ queda poco tiempo: me quedo con el de {BARRIDO[-1]['gb']} GB"); break
    if final is not None:
        # Si la memoria no va a alcanzar, mejor quedarse con el anterior que perder toda la corrida.
        # La cuenta es conservadora (supone que la RAM crece igual que la muestra); sin medición es nan y no frena.
        previsto = BARRIDO[-1]["ram_pico_gb"] * gb / BARRIDO[-1]["gb"]
        if previsto > 0.8 * RAM_TOTAL:
            log(f"   ⚠ el de {gb} GB pediría unos {previsto:.0f} GB de RAM y hay {RAM_TOTAL:.0f}: "
                f"me quedo con el de {BARRIDO[-1]['gb']} GB"); break
    tk, seg, ram_gb = entrenar(gb)
    bpt = {n: bytes_por_token(para_datos(tk), VAL[n]) for n in FUENTES}
    BARRIDO.append(dict(gb=gb, segundos=round(seg), ram_pico_gb=round(ram_gb, 2), vocab=tk.get_vocab_size(),
                        bytes_por_token={n: round(v, 4) for n, v in bpt.items()},
                        promedio=round(promedio_mezcla(bpt), 4), tronco=round(promedio_tronco(bpt), 4)))
    log(f"   {gb:>5} GB → {seg/60:5.1f} min | RAM pico {ram_gb:5.1f} GB | vocab {tk.get_vocab_size():,} | "
        f"{BARRIDO[-1]['promedio']:.3f} bytes/token en la muestra, {BARRIDO[-1]['tronco']:.3f} en el tronco")
    final = tk
TK, TKD = final, para_datos(final)

# Otras mezclas, todas con el tamaño chico del barrido: ¿cuánto mueve la compresión darle más o menos español?
COMPARA = {}
if COMPARAR_MEZCLAS:
    gb0 = BARRIDO[0]["gb"]
    COMPARA["final (refuerza español y código)"] = BARRIDO[0]["bytes_por_token"]
    for nombre_mezcla, dominios in MEZCLAS.items():
        if horas() > HORAS_MAX - 1.0:
            log("   ⏱ queda poco tiempo: no comparo más mezclas"); break
        tk_m, seg, _ = entrenar(gb0, pesos_de(dominios))
        COMPARA[nombre_mezcla] = {n: round(bytes_por_token(para_datos(tk_m), VAL[n]), 4) for n in FUENTES}
    log(f"\nMezclas de la muestra (bytes por token con {gb0} GB; más es mejor):")
    log("   " + f"{'mezcla':<36}" + "".join(f"{d:>14}" for d in TRONCO) + f"{'TRONCO':>10}")
    for nombre_mezcla, fila in COMPARA.items():
        d = por_dominio(fila)
        log("   " + f"{nombre_mezcla:<36}" + "".join(f"{d.get(k, float('nan')):>14.3f}" for k in TRONCO)
            + f"{promedio_tronco(fila):>10.3f}")
    log("   El tokenizador final usa la primera. Si otra le gana en TRONCO sin castigar al español, vale la pena verla.")

# ─── 7) pruebas ───
log("\nPruebas:")
PROBLEMAS = []
def ok(cond, msg, grave=True):
    log(f"   {'✅' if cond else ('❌' if grave else '⚠')} {msg}")
    if not cond and grave: PROBLEMAS.append(msg)

v = TK.get_vocab()
n_vocab = TK.get_vocab_size()
ok(n_vocab == VOCAB, f"vocabulario de {n_vocab:,} tokens (meta {VOCAB:,})", grave=not PRUEBA_RAPIDA)
ok(max(v.values()) < 65_536, "todos los ids caben en uint16")
ok([TK.id_to_token(i) for i in range(N_ESPECIALES)] == ESPECIALES, f"ids 0–{N_ESPECIALES - 1}: los tokens especiales, en orden")
ok(all(c in v for c in pre_tokenizers.ByteLevel.alphabet()), "los 256 bytes están en el vocabulario")

ESTRES = [
    "".join(chr(c) for c in range(128)),                                   # ASCII completo, con controles
    "".join(chr(c) for c in range(0x80, 0x800, 3)),                        # latín, griego, cirílico, armenio, hebreo, árabe, siríaco, thaana, n'ko
    "Ñandú, pingüino, ¿qué? ¡olé! café • naïve — “comillas” ‘simples’ … 10 °C, 5 €",
    "El corazón de la canción: «¿Dónde está el niño?» —preguntó—. Mañana será otro día, ¡ojalá!",
    "👨‍👩‍👧‍👦 🏳️‍🌈 🇨🇴 🏴\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F 👍🏽",
    "中文 日本語のテキスト 한국어 ภาษาไทย हिन्दी தமிழ் ქართული",
    "\U000F0000\U0010FFFD \uFFFD \uFEFF cero\u200bancho \u202Ertl",          # uso privado, BOM, ancho cero, RTL
    "e\u0301 vs é, ﬁ, Ⅻ, ①, ²",                                            # combinantes y compatibilidad (sin normalizar)
    "línea1\r\nlínea2\rlínea3\n\n\n\tTab\t\t" + " " * 1000 + "fin",
    "a" * 10_000,
    "def f(x):\n\tif x:\n\t\treturn x  # comentario\n" * 3 + "\x0c\x1b[31mrojo\x1b[0m\x00",
    "const f = async (x: number): Promise<string> => `${x ?? 0}`;\n<div class=\"a\" id='b'>&nbsp;</div>\n.a > .b::after { color: #fff; }",
    "SELECT u.id, COUNT(*) AS n FROM users u WHERE u.name LIKE '%ñ%' GROUP BY 1;\n$ pip install -U \"paquete>=1.2\" && echo $HOME",
    "١٢٣ ٤٥٦ ０１２ 12345678901234567890 3.14159e-10 0xDEADBEEF",
    " ".join(ESPECIALES[:20]) + " <|reserved_7|>",                          # en modo datos: texto normal
]
malos = [s for s in ESTRES if TKD.decode(TKD.encode(s).ids) != s]
ok(not malos, f"ida y vuelta exacta en las {len(ESTRES)} frases de estrés" + (f" (fallan {len(malos)})" if malos else ""))
todos_val = [t for n in FUENTES for t in VAL[n]]
malos_val = sum(TKD.decode(e.ids) != t for t, e in zip(todos_val, TKD.encode_batch(todos_val)))
ok(malos_val == 0, f"ida y vuelta exacta en {len(todos_val):,} documentos de val ({malos_val} fallan)")
control = sum(i < N_ESPECIALES for s in ESTRES for i in TKD.encode(s).ids)
ok(control == 0, "en modo datos, el texto crudo nunca produce tokens de control")
unicos = all(TK.encode(t, add_special_tokens=False).ids == [v[t]] for t in ESPECIALES)
ok(unicos, "en modo chat, cada token especial es un solo id")
ok([v[t] for t in ("<|endoftext|>", "<|pad|>", "<think>", "<|fim_prefix|>", "<|fim_pad|>")] == [0, 1, 4, 10, 13],
   "ids fijos del diseño: EOS 0, pad 1, <think> 4, FIM 10–13")
d = TK.decode(TK.encode("<|im_start|>assistant\n<think>pienso</think>hola<|im_end|>").ids, skip_special_tokens=True)
ok(d == "assistant\n<think>pienso</think>hola", f"<think> sobrevive a skip_special_tokens: {d!r}")
digitos = [TK.id_to_token(i) for i in TK.encode("12345").ids]
ok(digitos == list("12345"), f"los números se parten en dígitos: {digitos}")
ok("ĊĊ" in v, "'\\n\\n' (salto entre párrafos) es un solo token", grave=False)
ok("ĠĠĠ" in v, "la indentación tiene tokens propios (3 espacios + el de la palabra)", grave=False)
restos = [t for t, _ in NOMBRADOS if any(t in doc for doc in todos_val[:2000])]
ok(not restos, "el texto de la muestra no trae cadenas de control" + (f" (quedan: {restos})" if restos else ""))

# tokens que nunca salen (candidatos a "glitch tokens", como SolidGoldMagikarp en GPT-2/3).
# Los 256 bytes no cuentan: están para que nada se pierda, aunque casi no salgan.
cuenta = Counter()
lote = []
def _cuenta_lote():
    for e in TKD.encode_batch(lote, add_special_tokens=False): cuenta.update(e.ids)
    lote.clear()
for t in docs_muestra(MB_SIN_USO / 1e3):
    lote.append(t)
    if len(lote) >= 2000: _cuenta_lote()
lote.extend(todos_val); _cuenta_lote()
ALFABETO = set(pre_tokenizers.ByteLevel.alphabet())
SIN_USO = [i for i in range(N_ESPECIALES, n_vocab) if cuenta[i] == 0 and TK.id_to_token(i) not in ALFABETO]
ejemplos = [TK.decode([i]) for i in SIN_USO[:12]]
log(f"   ℹ {len(SIN_USO):,} tokens no salen en {MB_SIN_USO} MB de texto + los vals "
    f"({len(SIN_USO) / max(1, n_vocab - N_ESPECIALES):.2%}); ejemplos: {ejemplos}")

# ─── 8) compresión: el nuevo contra Ludwig 1 y las referencias ───
log("\nCompresión (bytes por token en los vals; más es mejor):")
REFS = {}
lud = [p for p in sorted(ENTRADA.rglob("tokenizer.json")) if (p.parent / "data_meta.json").exists()] if ENTRADA.exists() else []
if lud:
    try:
        REFS["Ludwig 1 (32K)"] = Tokenizer.from_file(str(lud[0]))
        log(f"   Ludwig 1: {lud[0]} | '\\n\\n' en su vocabulario: {'ĊĊ' in REFS['Ludwig 1 (32K)'].get_vocab()}")
    except Exception as e:
        log(f"   ⚠ no pude abrir el tokenizador de Ludwig 1 ({type(e).__name__})")
if not TEST:
    for nombre, repo in REFERENCIAS.items():
        try:
            REFS[nombre] = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", token=HF_TOKEN))
            REFS[nombre].no_truncation(); REFS[nombre].no_padding()
        except Exception as e:
            log(f"   ⚠ no pude bajar el tokenizador de {nombre} ({type(e).__name__}); queda fuera de la tabla")
TABLA = {"Nuevo (64K)": {n: bytes_por_token(TKD, VAL[n]) for n in FUENTES}}
for nombre, tk in REFS.items():
    TABLA[nombre] = {n: bytes_por_token(tk, VAL[n]) for n in FUENTES}
cols = list(TABLA)
log("   " + f"{'fuente':<19}" + "".join(f"{c:>16}" for c in cols))
for n in FUENTES:
    log("   " + f"{n:<19}" + "".join(f"{TABLA[c][n]:>16.3f}" for c in cols))
for dom in TRONCO:
    log("   " + f"{'· ' + dom:<19}" + "".join(f"{por_dominio(TABLA[c]).get(dom, float('nan')):>16.3f}" for c in cols))
log("   " + f"{'PROM. muestra':<19}" + "".join(f"{promedio_mezcla(TABLA[c]):>16.3f}" for c in cols))
log("   " + f"{'PROM. tronco':<19}" + "".join(f"{promedio_tronco(TABLA[c]):>16.3f}" for c in cols)
    + "   ← con la mezcla v3 del tronco: la que pone el costo")
for c in cols[1:]:
    peores = [n for n in FUENTES if TABLA["Nuevo (64K)"][n] < TABLA[c][n]]
    log(f"   vs {c}: el nuevo comprime menos en {len(peores)} de {len(FUENTES)} fuentes" +
        (f" ({', '.join(peores)})" if peores else ""))
def palabras(tk, textos):
    if not textos: return float("nan")
    n = sum(len(e.ids) for e in tk.encode_batch(textos, add_special_tokens=False))
    return n / max(1, sum(len(t.split()) for t in textos))
FERT = {c: {"inglés": palabras(TKD if c.startswith("Nuevo") else REFS[c], VAL.get("fineweb-edu", [])),
            "español": palabras(TKD if c.startswith("Nuevo") else REFS[c], VAL.get("fineweb2-hq-es", []))} for c in cols}
log("   tokens por palabra (inglés / español): " +
    " | ".join(f"{c} {FERT[c]['inglés']:.2f} / {FERT[c]['español']:.2f}" for c in cols))

# ─── 9) guardar ───
TK.save(str(SALIDA / "tokenizer.json"))
CHAT_TEMPLATE = ("{%- for m in messages -%}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
                 "{%- endfor -%}{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n' }}{%- endif -%}")
config = dict(
    tokenizer_class="PreTrainedTokenizerFast", model_max_length=16_384, clean_up_tokenization_spaces=False,
    bos_token=None, eos_token="<|endoftext|>", pad_token="<|pad|>", unk_token=None,
    add_bos_token=False, add_eos_token=False, chat_template=CHAT_TEMPLATE,
    added_tokens_decoder={str(i): dict(content=t, lstrip=False, normalized=False, rstrip=False, single_word=False,
                                       special=t not in VISIBLES) for i, t in enumerate(ESPECIALES)})
(SALIDA / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
(SALIDA / "special_tokens_map.json").write_text(json.dumps(dict(
    eos_token="<|endoftext|>", pad_token="<|pad|>",
    additional_special_tokens=[t for t, e in NOMBRADOS if e and t not in ("<|endoftext|>", "<|pad|>")]),
    ensure_ascii=False, indent=2), encoding="utf-8")
IDS = {t: i for i, t in enumerate(ESPECIALES[:len(NOMBRADOS)])}
(SALIDA / "cargar_tokenizador.py").write_text(f'''"""Cómo cargar el tokenizador 64K de Ludwig 2.

para_datos(): para TEXTO CRUDO (preentrenamiento). Ningún texto produce tokens de control:
    un "<|endoftext|>" o un "<think>" escritos en una página web quedan como texto normal.
    Los ids de control los pone el pipeline: EOS al final de cada documento, FIM, chat.
para_chat(): el normal, para SFT, servir y chatear. Ahí "<|im_start|>" sí es un token.
"""
import json
from tokenizers import Tokenizer

IDS = {json.dumps(IDS, ensure_ascii=False, indent=4)}
EOS, PAD = IDS["<|endoftext|>"], IDS["<|pad|>"]

def para_chat(ruta="tokenizer.json"):
    return Tokenizer.from_file(ruta)

def para_datos(ruta="tokenizer.json"):
    j = json.loads(open(ruta, encoding="utf-8").read())
    for a in j["added_tokens"]:
        a["special"] = True
    tk = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    tk.encode_special_tokens = True
    return tk

def documento(tk_datos, texto):
    """Un documento de preentrenamiento: sus tokens y el EOS al final."""
    return tk_datos.encode(texto, add_special_tokens=False).ids + [EOS]
''', encoding="utf-8")

sha1 = hashlib.sha1((SALIDA / "tokenizer.json").read_bytes()).hexdigest()
with open(SALIDA / "compresion.csv", "w", encoding="utf-8") as fh:
    fh.write("fuente,dominio," + ",".join(cols) + "\n")
    for n in FUENTES:
        fh.write(f"{n},{FUENTES[n]['dominio']}," + ",".join(f"{TABLA[c][n]:.4f}" for c in cols) + "\n")
    for dom in TRONCO:
        fh.write(f"DOMINIO,{dom}," + ",".join(f"{por_dominio(TABLA[c]).get(dom, float('nan')):.4f}" for c in cols) + "\n")
    fh.write("PROMEDIO,muestra," + ",".join(f"{promedio_mezcla(TABLA[c]):.4f}" for c in cols) + "\n")
    fh.write("PROMEDIO,tronco," + ",".join(f"{promedio_tronco(TABLA[c]):.4f}" for c in cols) + "\n")
with open(SALIDA / "mezclas.csv", "w", encoding="utf-8") as fh:
    fh.write("mezcla," + ",".join(TRONCO) + ",tronco\n")
    for nombre_mezcla, fila in COMPARA.items():
        dm = por_dominio(fila)
        fh.write(f"{nombre_mezcla}," + ",".join(f"{dm.get(k, float('nan')):.4f}" for k in TRONCO)
                 + f",{promedio_tronco(fila):.4f}\n")
meta = dict(
    version="tok64k-ludwig2-v1", script="v1.2", creado=time.strftime("%Y-%m-%d %H:%M"), prueba_rapida=PRUEBA_RAPIDA,
    semilla=SEMILLA, sha1_tokenizer_json=sha1, vocab=n_vocab, uint16=True,
    receta=dict(modelo="BPE a nivel de byte", regex=REGEX, regex_origen="Qwen2 (transformers y llama.cpp: 'qwen2')",
                normalizador=None, alfabeto_inicial="los 256 bytes", min_frequency=2,
                entrenamiento="train_from_iterator con documentos enteros (no línea por línea)",
                max_chars_doc=MAX_CHARS_DOC, cadenas_de_control="se quitan del texto de la muestra"),
    especiales=dict(ids=IDS, total=N_ESPECIALES, reservados=f"<|reserved_0|> … <|reserved_{N_ESPECIALES - len(NOMBRADOS) - 1}|>",
                    visibles_al_decodificar=sorted(VISIBLES),
                    regla_datos="el texto crudo nunca produce tokens de control (cargar_tokenizador.para_datos)"),
    mezcla={n: dict(repo=F["repo"], prefijo=F["prefijo"], dominio=F["dominio"], peso=round(F["peso"], 5),
                    licencia=F["licencia"], **STATS[n]) for n, F in FUENTES.items()},
    fuentes_que_fallaron=fallan, fuentes_no_muestreadas=NO_MUESTREADAS,
    codigo=dict(fuente="The Stack v1" if CODIGO_DE_THE_STACK else "Stack-Edu, solo licencia permisiva",
                software_heritage=dict(SWH)),
    regla_val="el val de cada fuente es su último archivo (o el último 10% de sus filas si tiene uno solo) "
              "y nunca entra al tokenizador",
    barrido=BARRIDO, compresion_bytes_por_token={c: {n: round(x, 4) for n, x in TABLA[c].items()} for c in cols},
    promedio_muestra={c: round(promedio_mezcla(TABLA[c]), 4) for c in cols},
    promedio_tronco={c: round(promedio_tronco(TABLA[c]), 4) for c in cols},
    por_dominio={c: {k: round(x, 4) for k, x in por_dominio(TABLA[c]).items()} for c in cols},
    tronco_v3=dict(partes_en_tokens=TRONCO, formula="suma de (parte del dominio × bytes por token del dominio)"),
    mezclas_comparadas=dict(gb=BARRIDO[0]["gb"], otras=MEZCLAS,
                            por_dominio={m: {k: round(x, 4) for k, x in por_dominio(f).items()} for m, f in COMPARA.items()},
                            tronco={m: round(promedio_tronco(f), 4) for m, f in COMPARA.items()}),
    promedio="muestra: bytes totales / tokens totales (armónico ponderado por bytes)",
    tokens_por_palabra={c: {k: round(x, 3) for k, x in FERT[c].items()} for c in cols},
    tokens_sin_uso=dict(n=len(SIN_USO), mb_de_texto=MB_SIN_USO, primeros=SIN_USO[:200]),
    problemas=PROBLEMAS, ram_total_gb=round(RAM_TOTAL, 1), horas=round(horas(), 2),
    versiones=dict(python=platform.python_version(), tokenizers=tokenizers.__version__,
                   pyarrow=None if TEST else pyarrow.__version__,
                   huggingface_hub=None if TEST else huggingface_hub.__version__))
(SALIDA / "tokenizer_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

dn = por_dominio(TABLA["Nuevo (64K)"])
resumen = [
    "═" * 70,
    f"TOKENIZADOR 64K DE LUDWIG 2 · {meta['creado']} · sha1 {sha1[:12]} · {n_vocab:,} tokens",
    f"Muestra: {sum(s['bytes'] for s in STATS.values())/1e6:,.0f} MB de {len(FUENTES)} fuentes" +
    (f" (fuera: {', '.join(fallan)})" if fallan else ""),
    "Barrido: " + " | ".join(f"{b['gb']} GB → {b['tronco']:.3f} B/tok en el tronco, {b['segundos']/60:.0f} min, "
                             f"RAM {b['ram_pico_gb']:.1f} GB" for b in BARRIDO),
    "Bytes/token por dominio: " + " | ".join(f"{k} {x:.3f}" for k, x in dn.items()),
    "Bytes/token, mezcla del tronco: " + " | ".join(f"{c} {promedio_tronco(TABLA[c]):.3f}" for c in cols),
    "Tokens por palabra (inglés / español): " + " | ".join(f"{c} {FERT[c]['inglés']:.2f} / {FERT[c]['español']:.2f}" for c in cols),
    "Mezclas (tronco): " + (" | ".join(f"{m} {promedio_tronco(f):.3f}" for m, f in COMPARA.items()) or "no se compararon"),
    f"Tokens sin uso en {MB_SIN_USO} MB: {len(SIN_USO):,}",
    ("✅ Todo bien" if not PROBLEMAS else f"❌ {len(PROBLEMAS)} problemas: " + " · ".join(PROBLEMAS)),
    f"Listo en {horas() * 60:.0f} min → {SALIDA}",
    "═" * 70]
(SALIDA / "reporte.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
log("\n" + "\n".join(resumen))
